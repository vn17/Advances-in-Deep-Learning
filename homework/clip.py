from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision as tv
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoProcessor, Trainer, TrainingArguments

from .base_vlm import BaseVLM
from .data import CaptionDataset, MultiChoiceQADataset

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load(model_name: str = "clip_model"):
    from pathlib import Path

    from peft import PeftModel

    model_path = Path(__file__).parent / model_name

    vlm = BaseVLM()
    vision_encoder = vlm.model.model.vision_model
    text_encoder = vlm.model.model.text_model
    clip = CLIP(vision_encoder, text_encoder)
    clip = PeftModel.from_pretrained(clip, model_path).to(device)

    clip.model.load_pretrained(model_path)
    clip.model.eval()
    if device == "cuda":
        clip = clip.to(dtype=torch.bfloat16)

    return clip


def clip_data_collator(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """
    Custom data collator for CLIP training.
    """
    # Get max sequence length
    max_length = max(f["input_ids"].shape[0] for f in features)

    def pad_tensor(tensor, pad_value):
        return torch.cat([tensor, torch.full((max_length - tensor.shape[0],), pad_value, dtype=tensor.dtype)])

    input_ids = torch.stack([pad_tensor(f["input_ids"], pad_value=processor.tokenizer.eos_token_id) for f in features])
    attention_mask = torch.stack([pad_tensor(f["attention_mask"], pad_value=0) for f in features])
    pixel_values = torch.stack([f["pixel_values"] for f in features])  # assume all are same shape
    labels = torch.stack([pad_tensor(f["labels"], pad_value=-100) for f in features])

    return {
        "input_ids": input_ids.long(),
        "attention_mask": attention_mask.long(),
        "pixel_values": pixel_values.float(),
        "labels": labels.long(),
    }


class CaptionDatasetForTraining(Dataset):
    def __init__(self, dataset: CaptionDataset, processor: AutoProcessor):
        self.dataset = dataset
        self.image_processor = tv.transforms.Compose(
            [
                tv.transforms.Resize(192),
                tv.transforms.RandomResizedCrop(192, scale=(0.5, 1.0)),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.dataset[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        pixel_values = self.image_processor(image)
        text = item["caption"] + self.processor.tokenizer.eos_token
        text_inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
        input_ids = text_inputs["input_ids"].squeeze(0).long()
        attention_mask = text_inputs["attention_mask"].squeeze(0)
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,  # placeholder to fit the collator
        }


class CLIP(nn.Module):
    def __init__(
        self, vision_encoder: nn.Module, text_encoder: nn.Module, proj_dim: int = 64, temperature: float = 0.07
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        # projection heads for vision and text
        # We'll infer hidden dims lazily on first forward pass, so create placeholders
        self.proj_dim = proj_dim
        self._vision_proj = None
        self._text_proj = None

        # temperature (use learnable log-temp)
        self.log_temp = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(1.0 / temperature))), dtype=torch.float32))

    def _lazy_init_projections(self, vision_feat: torch.Tensor, text_feat: torch.Tensor):
        # vision_feat: (B, ..., Dv) -> take last dim
        dv = vision_feat.shape[-1]
        dt = text_feat.shape[-1]
        if self._vision_proj is None:
            self._vision_proj = nn.Linear(dv, self.proj_dim)
            # register so parameters show up
            self.add_module("vision_projection", self._vision_proj)
        if self._text_proj is None:
            self._text_proj = nn.Linear(dt, self.proj_dim)
            self.add_module("text_projection", self._text_proj)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        out = self.vision_encoder(image)
        # prefer pooler_output if present
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            # mean pool
            return out.last_hidden_state.mean(dim=1)

        # fallback
        return out

    def encode_text(self, text: str) -> torch.Tensor:
        out = self.text_encoder(input_ids=text) if not isinstance(text, dict) else self.text_encoder(**text)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            # We expect text inputs to be token-level outputs; pool with attention if provided
            return out.last_hidden_state.mean(dim=1)

        return out

    def save_pretrained(self, save_directory: str, **kwargs):
        """Customize save method, save additional parameters"""

        additional_state_dict = {}
        for name, param in self.named_parameters():
            if "vision_encoder." in name or "text_encoder." in name:
                continue
            additional_state_dict[name] = param.data

        torch.save(additional_state_dict, Path(save_directory) / "additional_weights.pt")

    def load_pretrained(self, load_directory: str, **kwargs):
        """Customize load method, load projection additional parameters"""

        additional_weights_path = Path(load_directory) / "additional_weights.pt"
        if additional_weights_path.exists():
            additional_state_dict = torch.load(additional_weights_path, map_location="cpu")

            for name, param in self.named_parameters():
                if "vision_encoder." in name or "text_encoder." in name:
                    continue
                param.data = additional_state_dict[name]

    def set_trainable_parameters(self):
        for name, param in self.named_parameters():
            if "vision_encoder." in name or "text_encoder." in name:
                continue
            param.requires_grad = True

    def gradient_checkpointing_enable(self, **kwargs):
        """
        Enable gradient checkpointing for the vision and text backbones.
        (You don't need to touch this method)
        """
        self.vision_encoder.gradient_checkpointing_enable(**kwargs)
        self.text_encoder.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self):
        """
        Enable input require grads for the vision and text backbones.
        (You don't need to touch this method)
        """

        # Reference: https://discuss.huggingface.co/t/peft-lora-gpt-neox-backward-pass-failing/35641
        def make_inputs_require_grads(module, input, output):  # noqa: A002
            output.requires_grad_(True)

        self.vision_encoder.embeddings.register_forward_hook(make_inputs_require_grads)
        self.text_encoder.get_input_embeddings().register_forward_hook(make_inputs_require_grads)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None,
        labels: torch.Tensor = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for the CLIP model.
        Args:
            pixel_values: The pixel values of the image.
            input_ids: The input ids of the text.
            attention_mask: The attention mask of the text.
            labels: The labels for the text features.
            (NOTE: you don't need to use the variable `labels`, this is just for compatibility with the Trainer class)
            (Hint: refer to returned values of the __getitem__ method in the CaptionDatasetForTraining class)
        Returns:
            TODO: think about the what values should be returned
        """
        # pixel_values: (B, C, H, W) or (B, ...)
        # input_ids: either (N, L) where N == B (paired) or N may be > B (multi-candidates)

        # Encode images
        vision_out = self.vision_encoder(pixel_values)
        if hasattr(vision_out, "pooler_output") and vision_out.pooler_output is not None:
            vfeat = vision_out.pooler_output
        elif hasattr(vision_out, "last_hidden_state") and vision_out.last_hidden_state is not None:
            vfeat = vision_out.last_hidden_state.mean(dim=1)
        else:
            vfeat = vision_out

        # Encode texts
        # If attention_mask provided, call text_encoder with tensors
        if isinstance(input_ids, torch.Tensor):
            text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            # assume dict-like (already prepared)
            text_out = self.text_encoder(**input_ids)

        if hasattr(text_out, "pooler_output") and text_out.pooler_output is not None:
            tfeat = text_out.pooler_output
        elif hasattr(text_out, "last_hidden_state") and text_out.last_hidden_state is not None:
            # mean pool using attention mask if available
            if attention_mask is not None:
                mask = attention_mask.float().unsqueeze(-1)
                tfeat = (text_out.last_hidden_state * mask).sum(dim=1) / (mask.sum(dim=1).clamp(min=1e-9))
            else:
                tfeat = text_out.last_hidden_state.mean(dim=1)
        else:
            tfeat = text_out

        # lazy init projection layers
        self._lazy_init_projections(vfeat, tfeat)

        vproj = self._vision_proj(vfeat)
        tproj = self._text_proj(tfeat)

        # normalize
        vnorm = vproj / (vproj.norm(dim=-1, keepdim=True).clamp(min=1e-9))
        tnorm = tproj / (tproj.norm(dim=-1, keepdim=True).clamp(min=1e-9))

        # temperature
        temperature = torch.exp(-self.log_temp)

        # logits: (num_images, num_texts)
        logits = torch.matmul(vnorm, tnorm.T) / temperature

        return vnorm, tnorm, logits


def compute_clip_loss(
    outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    labels: torch.Tensor,
    num_items_in_batch: int | None = None,
) -> torch.Tensor:
    """
    Compute the loss for the CLIP model.
    Args:
        outputs: A tuple containing the outputs of CLIP.forward().
        labels: The labels for the text features.
        (NOTE: you don't need to use the variable `labels`, this is just for compatibility with the Trainer class)
        num_items_in_batch: The number of items in the batch.
        (NOTE: you don't need to use the variable `num_items_in_batch`, this is just for compatibility with Trainer)
    Returns:
        The loss for the CLIP model.
    """
    # outputs: (vision_feat, text_feat, logits)
    if not isinstance(outputs, (list, tuple)) or len(outputs) < 3:
        raise ValueError("Expected outputs tuple (vfeat, tfeat, logits)")

    _, _, logits = outputs

    # If logits is 2D square and we assume i-th image matches i-th text
    device = logits.device
    n_i, n_t = logits.shape

    labels_idx = torch.arange(n_i, device=device)

    # image-to-text loss
    loss_i2t = nn.functional.cross_entropy(logits, labels_idx)
    # text-to-image loss
    loss_t2i = nn.functional.cross_entropy(logits.T, labels_idx)

    return (loss_i2t + loss_t2i) / 2.0


def get_target_modules_for_lora(model: nn.Module) -> list[str]:
    target_modules = []
    for name, module in model.named_modules():
        # if isinstance(module, nn.Linear) and ("vision_encoder" in name and "projection" not in name):
        if (
            isinstance(module, nn.Linear)
            and ("vision_encoder" in name or "text_encoder" in name)
            and "projection" not in name
        ):
            target_modules.append(name)

    return target_modules


def train(
    data_dir: Path | None = None,
    output_dir: str = "clip",
    num_train_epochs: float = 0.05,  # for debugging purpose, increase this once the dry run works
    per_device_train_batch_size: int = 1024,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 5e-4,
    num_workers: int = 16,
):
    vlm = BaseVLM()

    output_dir = Path(__file__).parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize TensorBoard writer
    tensorboard_dir = output_dir / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # Initialize model and processor
    vision_encoder = vlm.model.model.vision_model
    text_encoder = vlm.model.model.text_model
    model = CLIP(vision_encoder, text_encoder).to(device).bfloat16()
    model.set_trainable_parameters()

    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.0,
        # target_modules="all-linear",
        target_modules=get_target_modules_for_lora(model),
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.to(device)
    model.train()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # load dataset
    train_dataset = CaptionDataset("train", data_dir)
    train_dataset = CaptionDatasetForTraining(train_dataset, processor)

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=learning_rate,
        bf16=True if device == "cuda" else False,
        logging_steps=1,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        label_names=["labels"],
        dataloader_num_workers=num_workers,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=clip_data_collator,
        compute_loss_func=compute_clip_loss,
    )

    trainer.train()

    # save model
    trainer.save_model(output_dir)
    model.model.save_pretrained(output_dir)

    writer.close()

    return model, processor


def demo_train():
    train(
        train_dataset_name="train_demo",
        output_dir="demo_clip",
        num_train_epochs=1,
        per_device_train_batch_size=2,
        num_workers=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-8,
    )


def test(ckpt_path: str, val_dataset: str = "valid_grader"):
    import tqdm

    testset = MultiChoiceQADataset(val_dataset)

    clip = load(ckpt_path)
    clip = clip.model.to(device)

    image_processor = tv.transforms.Compose(
        [
            tv.transforms.Resize(192),
            tv.transforms.CenterCrop(192),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    correct_count = 0
    total_count = 0

    for pair in tqdm.tqdm(testset):
        image = Image.open(pair["image_path"]).convert("RGB")
        pixel_values = image_processor(image).unsqueeze(0).to(device).bfloat16()
        text_inputs = processor(
            text=[s + processor.tokenizer.eos_token for s in pair["candidates"]],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = text_inputs["input_ids"].long().to(device)
        attention_mask = text_inputs["attention_mask"].to(device)
        vision_feature, text_feature, _ = clip(pixel_values, input_ids, attention_mask)
        prediction = torch.matmul(vision_feature, text_feature.T).argmax(dim=-1)
        if prediction == pair["correct_index"]:
            correct_count += 1
        total_count += 1

    print(f"Accuracy: {correct_count / total_count}")


def main():
    from fire import Fire

    Fire({"train": train, "test": test})


if __name__ == "__main__":
    main()
