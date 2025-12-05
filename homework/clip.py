from pathlib import Path
import random
from typing import Any

import torch
import torch.nn as nn
import torchvision as tv
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoProcessor, Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback

from .base_vlm import BaseVLM
from .data import CaptionDataset, MultiChoiceQADataset

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class SaveAdditionalWeightsCallback(TrainerCallback):
    """Callback to save additional_weights.pt at each checkpoint and end of training."""
    def __init__(self, output_dir):
        self.output_dir = output_dir
    
    def on_save(self, args, state, control, **kwargs):
        """Called whenever a checkpoint is saved"""
        model = kwargs.get('model')
        if model is not None:
            if hasattr(model, 'module'):
                clip_model = model.module
            else:
                clip_model = model.model if hasattr(model, 'model') else model
            
            if hasattr(clip_model, 'save_pretrained'):
                clip_model.save_pretrained(self.output_dir)
                print(f"✓ Saved additional weights to {self.output_dir}")
    
    def on_train_end(self, args, state, control, **kwargs):
        """Called at end of training"""
        self.on_save(args, state, control, **kwargs)


def force_lazy_init(model, device):
    """Run one small forward pass to initialize lazy projection layers."""
    with torch.no_grad():
        # Get expected image size from vision encoder config
        try:
            vision_config = model.vision_encoder.config if hasattr(model.vision_encoder, 'config') else None
            img_size = vision_config.image_size if vision_config and hasattr(vision_config, 'image_size') else 224
        except:
            img_size = 224
        
        print(f"Force initializing projection layers with image size {img_size}x{img_size}")
        
        # Create dummy inputs
        img = torch.randn(1, 3, img_size, img_size).to(device)
        
        # Use real tokenizer to produce valid input_ids and attention_mask
        sample = processor.tokenizer(
            "hello world",
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        ids = sample["input_ids"].to(device)
        mask = sample["attention_mask"].to(device)

        try:
            _ = model(img, ids, mask)
            print("✓ Projection layers initialized successfully")
        except Exception as e:
            print(f"⚠ force_lazy_init failed (non-fatal): {repr(e)}")


def load(model_name: str = "clip_model"):
    """Load a trained CLIP model from checkpoint."""
    model_path = Path(__file__).parent / model_name

    vlm = BaseVLM()
    vision_encoder = vlm.model.model.vision_model
    text_encoder = vlm.model.model.text_model
    clip_net = CLIP(vision_encoder, text_encoder)
    adapter_config = model_path / "adapter_config.json"

    if model_path.exists() and adapter_config.exists():
        try:
            # Load PEFT adapter
            clip = PeftModel.from_pretrained(clip_net, model_path)
            print(f"✓ Loaded PEFT adapter from {model_path}")
            
            # Force lazy init to create projection layers
            clip = clip.to(device)
            if device == "cuda":
                clip = clip.to(dtype=torch.bfloat16)
            force_lazy_init(clip, device)
            
            # Load additional weights (projections, temperature) - FAIL if not available
            additional_weights_path = model_path / "additional_weights.pt"
            if not additional_weights_path.exists():
                raise FileNotFoundError(
                    f"Additional weights file not found at {additional_weights_path}. "
                    f"This file is required for proper model loading. "
                    f"Please ensure the model was trained and saved correctly."
                )
            
            clip.model.load_pretrained(model_path)
            print(f"✓ Loaded additional weights from {model_path}")
            
            clip.model.eval()
            return clip
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")

    raise FileNotFoundError(f"No adapter found at {model_path}. Train first before testing.")


def clip_data_collator(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Custom data collator for CLIP training."""
    max_length = max(f["input_ids"].shape[0] for f in features)

    def pad_tensor(tensor, pad_value):
        return torch.cat([tensor, torch.full((max_length - tensor.shape[0],), pad_value, dtype=tensor.dtype)])

    input_ids = torch.stack([pad_tensor(f["input_ids"], pad_value=processor.tokenizer.eos_token_id) for f in features])
    attention_mask = torch.stack([pad_tensor(f["attention_mask"], pad_value=0) for f in features])
    pixel_values = torch.stack([f["pixel_values"] for f in features])
    labels = torch.stack([pad_tensor(f["labels"], pad_value=-100) for f in features])

    return {
        "input_ids": input_ids.long(),
        "attention_mask": attention_mask.long(),
        "pixel_values": pixel_values.float(),
        "labels": labels.long(),
    }


class CaptionDatasetForTraining(Dataset):
    """Dataset wrapper that applies augmentation and tokenization for CLIP training."""
    def __init__(self, dataset: CaptionDataset, processor: AutoProcessor):
        self.dataset = dataset
        self.image_processor = tv.transforms.Compose([
            tv.transforms.Resize(224),
            tv.transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.dataset[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        pixel_values = self.image_processor(image)
        
        # Captions are already individual fragments from generate_captions.py
        # No need to split again!
        text = item["caption"] + self.processor.tokenizer.eos_token

        text_inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True)
        input_ids = text_inputs["input_ids"].squeeze(0).long()
        attention_mask = text_inputs["attention_mask"].squeeze(0)
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,
        }


class CLIP(nn.Module):
    """
    CLIP model with lazy projection layer initialization.
    Uses vision and text encoders from a base VLM and adds projection heads.
    """
    def __init__(self, vision_encoder: nn.Module, text_encoder: nn.Module, proj_dim: int = 64, temperature: float = 0.07):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.proj_dim = proj_dim
        self._vision_proj = None
        self._text_proj = None
        
        # Learnable temperature parameter (stored as log for numerical stability)
        self.log_temp = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(1.0 / temperature))), dtype=torch.float32))

    def _lazy_init_projections(self, vision_feat: torch.Tensor, text_feat: torch.Tensor):
        """Initialize projection layers on first forward pass based on feature dimensions."""
        dv = vision_feat.shape[-1]
        dt = text_feat.shape[-1]
        
        # Get device and dtype from existing model parameters
        device = next(self.vision_encoder.parameters()).device
        dtype = next(self.vision_encoder.parameters()).dtype
        
        if self._vision_proj is None:
            self._vision_proj = nn.Linear(dv, self.proj_dim).to(device=device, dtype=dtype)
            self.add_module("vision_projection", self._vision_proj)
        if self._text_proj is None:
            self._text_proj = nn.Linear(dt, self.proj_dim).to(device=device, dtype=dtype)
            self.add_module("text_projection", self._text_proj)

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode image to feature vector."""
        out = self.vision_encoder(image)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            return out.last_hidden_state.mean(dim=1)
        return out

    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to feature vector."""
        out = self.text_encoder(input_ids=text) if not isinstance(text, dict) else self.text_encoder(**text)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            return out.pooler_output
        if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            return out.last_hidden_state.mean(dim=1)
        return out

    def save_pretrained(self, save_directory: str, **kwargs):
        """Save custom parameters (projections and temperature) to additional_weights.pt"""
        additional_state_dict = {}
        for name, param in self.named_parameters():
            if "vision_encoder." in name or "text_encoder." in name:
                continue
            additional_state_dict[name] = param.data
        torch.save(additional_state_dict, Path(save_directory) / "additional_weights.pt")

    def load_pretrained(self, load_directory: str, **kwargs):
        """Load custom parameters (projections and temperature) from additional_weights.pt"""
        additional_weights_path = Path(load_directory) / "additional_weights.pt"
        if additional_weights_path.exists():
            device = next(self.parameters()).device
            additional_state_dict = torch.load(additional_weights_path, map_location=device)
            for name, param in self.named_parameters():
                if "vision_encoder." in name or "text_encoder." in name:
                    continue
                if name in additional_state_dict:
                    param.data = additional_state_dict[name].to(device=device, dtype=param.dtype)

    def set_trainable_parameters(self):
        """Set custom parameters (projections and temperature) as trainable."""
        for name, param in self.named_parameters():
            if "vision_encoder." in name or "text_encoder." in name:
                continue
            param.requires_grad = True

    def gradient_checkpointing_enable(self, **kwargs):
        """Enable gradient checkpointing for memory efficiency."""
        self.vision_encoder.gradient_checkpointing_enable(**kwargs)
        self.text_encoder.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self):
        """Enable input gradients for PEFT compatibility."""
        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)
        self.vision_encoder.embeddings.register_forward_hook(make_inputs_require_grads)
        self.text_encoder.get_input_embeddings().register_forward_hook(make_inputs_require_grads)

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, labels: torch.Tensor = None, **kwargs):
        """
        Forward pass for CLIP model.
        
        Returns:
            tuple: (vision_normalized, text_normalized, logits)
                - vision_normalized: normalized vision embeddings (B, proj_dim)
                - text_normalized: normalized text embeddings (B, proj_dim)
                - logits: similarity matrix (B_images, B_texts)
        """
        target_dtype = next(self.vision_encoder.parameters()).dtype
        pixel_values = pixel_values.to(dtype=target_dtype)

        # Encode images
        vision_out = self.vision_encoder(pixel_values)
        if hasattr(vision_out, "pooler_output") and vision_out.pooler_output is not None:
            vfeat = vision_out.pooler_output
        elif hasattr(vision_out, "last_hidden_state") and vision_out.last_hidden_state is not None:
            vfeat = vision_out.last_hidden_state.mean(dim=1)
        else:
            vfeat = vision_out

        # Encode texts
        if isinstance(input_ids, torch.Tensor):
            text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            text_out = self.text_encoder(**input_ids)

        if hasattr(text_out, "pooler_output") and text_out.pooler_output is not None:
            tfeat = text_out.pooler_output
        elif hasattr(text_out, "last_hidden_state") and text_out.last_hidden_state is not None:
            if attention_mask is not None:
                mask = attention_mask.float().unsqueeze(-1)
                tfeat = (text_out.last_hidden_state * mask).sum(dim=1) / (mask.sum(dim=1).clamp(min=1e-9))
            else:
                tfeat = text_out.last_hidden_state.mean(dim=1)
        else:
            tfeat = text_out

        # Ensure features match model dtype
        vfeat = vfeat.to(dtype=target_dtype)
        tfeat = tfeat.to(dtype=target_dtype)

        # Initialize projection layers if needed
        self._lazy_init_projections(vfeat, tfeat)

        # Project to shared embedding space
        vproj = self._vision_proj(vfeat)
        tproj = self._text_proj(tfeat)

        # Normalize embeddings
        vnorm = vproj / (vproj.norm(dim=-1, keepdim=True).clamp(min=1e-9))
        tnorm = tproj / (tproj.norm(dim=-1, keepdim=True).clamp(min=1e-9))

        # Compute similarity with learned temperature
        temperature = torch.exp(-self.log_temp)
        logits = torch.matmul(vnorm, tnorm.T) / temperature

        return vnorm, tnorm, logits


def compute_clip_loss(outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor], labels: torch.Tensor, num_items_in_batch: int | None = None) -> torch.Tensor:
    """
    Compute contrastive loss for CLIP model.
    Assumes i-th image matches i-th text (diagonal matching).
    
    Args:
        outputs: Tuple of (vision_norm, text_norm, logits)
        labels: Unused (kept for Trainer compatibility)
        num_items_in_batch: Unused (kept for Trainer compatibility)
        
    Returns:
        Combined image-to-text and text-to-image loss
    """
    _, _, logits = outputs
    device = logits.device
    n_i, n_t = logits.shape
    
    # Ground truth: i-th image matches i-th text
    labels_idx = torch.arange(n_i, device=device)
    
    # Image-to-text loss: for each image, predict matching text
    loss_i2t = nn.functional.cross_entropy(logits, labels_idx)
    
    # Text-to-image loss: for each text, predict matching image
    loss_t2i = nn.functional.cross_entropy(logits.T, labels_idx)
    
    return (loss_i2t + loss_t2i) / 2.0


def get_target_modules_for_lora(model: nn.Module) -> list[str]:
    """Identify linear layers in encoders for LoRA adaptation."""
    target_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and ("vision_encoder" in name or "text_encoder" in name) and "projection" not in name:
            target_modules.append(name)
    return target_modules


def train(
    data_dir: Path | None = None,
    output_dir: str = "clip_model",
    num_train_epochs: float = 1,
    per_device_train_batch_size: int = 128, # Increased batch size (CLIP needs large batches)
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 1e-4,
    num_workers: int = 4,
):
    vlm = BaseVLM()

    output_dir = Path(__file__).parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize TensorBoard writer
    tensorboard_dir = output_dir / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # Initialize model
    vision_encoder = vlm.model.model.vision_model
    text_encoder = vlm.model.model.text_model
    
    # 1. INCREASE DIMENSION: 64 is too small. Use 512 or 768.
    model = CLIP(vision_encoder, text_encoder, proj_dim=512).to(device).bfloat16()

    # --- CRITICAL FIX START: FORCE INITIALIZATION ---
    print("⚡ Running dummy pass to initialize projection layers...")
    dummy_image = torch.randn(1, 3, 384, 384).to(device).bfloat16() # Adjust size if needed (e.g. 512)
    # Create a simple dummy text input
    dummy_text = processor.tokenizer("dummy", return_tensors="pt").to(device)
    
    with torch.no_grad():
        # This triggers _lazy_init_projections immediately
        model(
            pixel_values=dummy_image, 
            input_ids=dummy_text.input_ids, 
            attention_mask=dummy_text.attention_mask
        )
    print("✅ Projection layers initialized.")
    # --- CRITICAL FIX END ---

    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=16, # Increased R for better expressivity
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=get_target_modules_for_lora(model),
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    
    # Now set trainable parameters. Since layers exist, this will actually work.
    model.set_trainable_parameters()
    
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
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        dataloader_pin_memory=True,
        remove_unused_columns=False, # IMPORTANT for custom collators
        label_names=["labels"],
        dataloader_num_workers=num_workers,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=clip_data_collator,
        compute_loss_func=compute_clip_loss,
        callbacks=[SaveAdditionalWeightsCallback(output_dir)],
    )

    trainer.train(resume_from_checkpoint=False) # Recommend False to start fresh with fixed weights

    trainer.save_model(output_dir)
    writer.close()

    return model, processor


def test(ckpt_path: str = "clip_model", val_dataset: str = "valid_grader", show_failures: bool = True, max_failures: int = 20):
    """
    Test CLIP model on multiple-choice QA dataset.
    
    Args:
        ckpt_path: Path to model checkpoint
        val_dataset: Validation dataset split name
        show_failures: Whether to print failing samples
        max_failures: Maximum number of failures to display
    """
    import tqdm

    testset = MultiChoiceQADataset(val_dataset)
    clip = load(ckpt_path)
    clip = clip.to(device)
    clip.eval()

    image_processor = tv.transforms.Compose([
        tv.transforms.Resize(224),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    correct_count = 0
    total_count = 0
    failures = []

    print(f"\n{'='*60}")
    print(f"Testing on {len(testset)} samples")
    print(f"{'='*60}\n")

    for idx, pair in enumerate(tqdm.tqdm(testset)):
        image = Image.open(pair["image_path"]).convert("RGB")
        pixel_values = image_processor(image).unsqueeze(0).to(device)
        target_dtype = next(clip.model.vision_encoder.parameters()).dtype
        pixel_values = pixel_values.to(dtype=target_dtype)

        # Use ORIGINAL candidates - don't filter!
        text_inputs = processor(
            text=[s + processor.tokenizer.eos_token for s in pair["candidates"]],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = text_inputs["input_ids"].long().to(device)
        attention_mask = text_inputs["attention_mask"].to(device)

        with torch.no_grad():
            _, _, logits = clip(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)

        # Pick candidate with highest similarity
        prediction = logits.argmax(dim=-1).item()
        
        if prediction == pair["correct_index"]:
            correct_count += 1
        else:
            # Store failure information
            similarities = logits.squeeze(0).cpu().tolist()
            failures.append({
                'sample_idx': idx,
                'image_path': pair["image_path"],
                'candidates': pair["candidates"],
                'correct_index': pair["correct_index"],
                'predicted_index': prediction,
                'similarities': similarities,
            })
        
        total_count += 1

    accuracy = correct_count / total_count
    print(f"\n{'='*60}")
    print(f"Accuracy: {accuracy:.4f} ({correct_count}/{total_count})")
    print(f"Failed: {len(failures)}/{total_count}")
    print(f"{'='*60}\n")
    
    # Print failing samples
    if show_failures and failures:
        print(f"\n{'='*60}")
        print(f"FAILING SAMPLES (showing up to {max_failures})")
        print(f"{'='*60}\n")
        
        for i, failure in enumerate(failures[:max_failures]):
            print(f"\n{'─'*60}")
            print(f"Failure #{i+1} (Sample {failure['sample_idx']})")
            print(f"{'─'*60}")
            print(f"Image: {failure['image_path']}")
            print(f"\nCandidates (with similarity scores):")
            
            for j, (candidate, sim) in enumerate(zip(failure['candidates'], failure['similarities'])):
                marker = ""
                if j == failure['correct_index']:
                    marker = " ✓ CORRECT"
                elif j == failure['predicted_index']:
                    marker = " ✗ PREDICTED"
                
                print(f"  [{j}] (sim: {sim:6.3f}) {candidate}{marker}")
            
            print(f"\nCorrect: [{failure['correct_index']}] {failure['candidates'][failure['correct_index']]}")
            print(f"Predicted: [{failure['predicted_index']}] {failure['candidates'][failure['predicted_index']]}")
        
        if len(failures) > max_failures:
            print(f"\n... and {len(failures) - max_failures} more failures not shown")
        
        print(f"\n{'='*60}\n")
    
    return accuracy, failures


def main():
    from fire import Fire
    Fire({"train": train, "test": test})


if __name__ == "__main__":
    main()