from pathlib import Path
import random
from typing import Any
import numpy as np
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
        model = kwargs.get('model')
        if model is not None:
            if hasattr(model, 'module'):
                clip_model = model.module.model
            else:
                clip_model = model.model if hasattr(model, 'model') else model
            
            if hasattr(clip_model, 'save_pretrained'):
                clip_model.save_pretrained(self.output_dir)
                print(f"✓ Saved additional weights to {self.output_dir}")
    
    def on_train_end(self, args, state, control, **kwargs):
        self.on_save(args, state, control, **kwargs)


def force_lazy_init(model, device):
    """Run one small forward pass to initialize lazy projection layers."""
    with torch.no_grad():
        try:
            vision_config = model.vision_encoder.config if hasattr(model.vision_encoder, 'config') else None
            img_size = vision_config.image_size if vision_config and hasattr(vision_config, 'image_size') else 224
        except:
            img_size = 224
        
        print(f"Force initializing projection layers with image size {img_size}x{img_size}")
        
        img = torch.randn(1, 3, img_size, img_size).to(device)
        
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
            clip = PeftModel.from_pretrained(clip_net, model_path)
            print(f"✓ Loaded PEFT adapter from {model_path}")
            
            clip = clip.to(device)
            if device == "cuda":
                clip = clip.to(dtype=torch.bfloat16)
            force_lazy_init(clip, device)
            
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
        # IMPROVED: More aggressive augmentation for better generalization
        self.image_processor = tv.transforms.Compose([
            tv.transforms.Resize(256),  # Larger initial size
            tv.transforms.RandomResizedCrop(224, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
            tv.transforms.RandomHorizontalFlip(p=0.5),
            tv.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            tv.transforms.RandomGrayscale(p=0.1),
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
        
        text = item["caption"]
        if not text.endswith(self.processor.tokenizer.eos_token):
            text = text + self.processor.tokenizer.eos_token

        text_inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True, max_length=77)
        input_ids = text_inputs["input_ids"].squeeze(0).long()
        attention_mask = text_inputs["attention_mask"].squeeze(0)
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,
        }


class CLIP(nn.Module):
    def __init__(self, vision_encoder: nn.Module, text_encoder: nn.Module, proj_dim: int = 512):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.proj_dim = proj_dim
        self._vision_proj = None
        self._text_proj = None
        
        # Temperature parameter (learnable)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
    def _lazy_init_projections(self, vision_feat: torch.Tensor, text_feat: torch.Tensor):
        if self._vision_proj is not None and self._text_proj is not None:
            return 
            
        dv = vision_feat.shape[-1]
        dt = text_feat.shape[-1]
        
        device = next(self.vision_encoder.parameters()).device
        target_dtype = next(self.vision_encoder.parameters()).dtype
        
        if self._vision_proj is None:
            # Multi-layer projection for vision (better capacity)
            self._vision_proj = nn.Sequential(
                nn.Linear(dv, dv, bias=False),
                nn.GELU(),
                nn.Linear(dv, self.proj_dim, bias=False)
            ).to(device=device, dtype=torch.float32)
            
            # Initialize properly
            for m in self._vision_proj.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight)
            
            self._vision_proj = self._vision_proj.to(dtype=target_dtype)
            self.add_module("vision_projection", self._vision_proj)
            print(f"✓ Initialized vision projection: {dv} -> {self.proj_dim} (2-layer)")
            
        if self._text_proj is None:
            # Multi-layer projection for text
            self._text_proj = nn.Sequential(
                nn.Linear(dt, dt, bias=False),
                nn.GELU(),
                nn.Linear(dt, self.proj_dim, bias=False)
            ).to(device=device, dtype=torch.float32)
            
            for m in self._text_proj.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight)
            
            self._text_proj = self._text_proj.to(dtype=target_dtype)
            self.add_module("text_projection", self._text_proj)
            print(f"✓ Initialized text projection: {dt} -> {self.proj_dim} (2-layer)")

    def save_pretrained(self, save_directory: str, **kwargs):
        additional_state_dict = {}
        for name, param in self.named_parameters():
            if "vision_encoder." in name or "text_encoder." in name:
                continue
            additional_state_dict[name] = param.data
        
        save_path = Path(save_directory) / "additional_weights.pt"
        torch.save(additional_state_dict, save_path)

    def load_pretrained(self, load_directory: str, **kwargs):
        additional_weights_path = Path(load_directory) / "additional_weights.pt"
        if additional_weights_path.exists():
            device = next(self.parameters()).device
            additional_state_dict = torch.load(additional_weights_path, map_location=device, weights_only=True)
            
            for name, param in self.named_parameters():
                if "vision_encoder." in name or "text_encoder." in name:
                    continue
                if name in additional_state_dict:
                    param.data = additional_state_dict[name].to(device=device, dtype=param.dtype)
            print(f"✓ Loaded weights from {additional_weights_path}")

    def set_trainable_parameters(self):
        for name, param in self.named_parameters():
            if "vision_encoder." in name or "text_encoder." in name:
                continue
            param.requires_grad = True

    def gradient_checkpointing_enable(self, **kwargs):
        self.vision_encoder.gradient_checkpointing_enable(**kwargs)
        self.text_encoder.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self):
        def make_inputs_require_grads(module, input, output):
            output.requires_grad_(True)
        self.vision_encoder.embeddings.register_forward_hook(make_inputs_require_grads)
        self.text_encoder.get_input_embeddings().register_forward_hook(make_inputs_require_grads)

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, labels: torch.Tensor = None, **kwargs):
        target_dtype = next(self.vision_encoder.parameters()).dtype
        pixel_values = pixel_values.to(dtype=target_dtype)

        # 1. Vision Forward - use CLS token if available
        vision_out = self.vision_encoder(pixel_values)
        if hasattr(vision_out, "pooler_output") and vision_out.pooler_output is not None:
            vfeat = vision_out.pooler_output
        elif hasattr(vision_out, "last_hidden_state"):
            # Use CLS token (first token) for ViT
            vfeat = vision_out.last_hidden_state[:, 0]
        else:
            vfeat = vision_out

        # 2. Text Forward
        if isinstance(input_ids, torch.Tensor):
            text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        else:
            text_out = self.text_encoder(**input_ids)

        # 3. Text Pooling - Mean pooling with attention mask
        if hasattr(text_out, "last_hidden_state"):
            hidden_states = text_out.last_hidden_state
            
            if attention_mask is not None:
                # Expand mask and apply
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                tfeat = sum_embeddings / sum_mask
            else:
                tfeat = hidden_states.mean(dim=1)
        elif hasattr(text_out, "pooler_output"):
            tfeat = text_out.pooler_output
        else:
            tfeat = text_out

        # Ensure dtypes match
        vfeat = vfeat.to(dtype=target_dtype)
        tfeat = tfeat.to(dtype=target_dtype)

        self._lazy_init_projections(vfeat, tfeat)

        # Project with multi-layer projections
        vproj = self._vision_proj(vfeat)
        tproj = self._text_proj(tfeat)
        
        # L2 normalize
        vnorm = vproj / (vproj.norm(dim=-1, keepdim=True).clamp(min=1e-9))
        tnorm = tproj / (tproj.norm(dim=-1, keepdim=True).clamp(min=1e-9))

        # Compute logits with learned temperature
        logit_scale = self.logit_scale.exp().clamp(max=100)
        logits = torch.matmul(vnorm, tnorm.T) * logit_scale

        return vnorm, tnorm, logits


def compute_clip_loss(outputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor], labels: torch.Tensor, num_items_in_batch: int | None = None) -> torch.Tensor:
    """Compute symmetric contrastive loss for CLIP."""
    _, _, logits = outputs
    device = logits.device
    n = logits.shape[0]
    
    # Create labels (diagonal should match)
    labels_idx = torch.arange(n, device=device)
    
    # Symmetric loss
    loss_i2t = nn.functional.cross_entropy(logits, labels_idx)
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
    num_train_epochs: float = 10,
    per_device_train_batch_size: int = 64,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    num_workers: int = 8,
    warmup_ratio: float = 0.1,
):
    """Train CLIP model with improved hyperparameters."""
    vlm = BaseVLM()

    output_dir = Path(__file__).parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_dir = output_dir / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # Initialize model with LARGER projection dimension
    vision_encoder = vlm.model.model.vision_model
    text_encoder = vlm.model.model.text_model
    
    model = CLIP(vision_encoder, text_encoder, proj_dim=512).to(device)
    
    if device == "cuda":
        model = model.bfloat16()

    # Force initialization
    print("\n" + "="*60)
    print("Initializing projection layers...")
    print("="*60)
    dummy_image = torch.randn(1, 3, 224, 224).to(device)
    if device == "cuda":
        dummy_image = dummy_image.bfloat16()
    
    dummy_text = processor.tokenizer("dummy", return_tensors="pt").to(device)
    
    with torch.no_grad():
        model(
            pixel_values=dummy_image, 
            input_ids=dummy_text.input_ids, 
            attention_mask=dummy_text.attention_mask
        )
    print("="*60 + "\n")

    # LoRA configuration - MORE aggressive adaptation
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=128,  # Higher rank
        lora_alpha=256,  # Higher alpha
        lora_dropout=0.1,
        target_modules=get_target_modules_for_lora(model),
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    
    model.set_trainable_parameters()
    
    print("\n" + "="*60)
    model.print_trainable_parameters()
    print("="*60 + "\n")
    
    model.to(device)
    model.train()
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # Load dataset
    train_dataset = CaptionDataset("train", data_dir)
    train_dataset = CaptionDatasetForTraining(train_dataset, processor)
    
    effective_batch_size = per_device_train_batch_size * gradient_accumulation_steps
    
    print("="*60)
    print(f"IMPROVED Training Configuration:")
    print(f"  Dataset size: {len(train_dataset)}")
    print(f"  Batch size per device: {per_device_train_batch_size}")
    print(f"  Gradient accumulation: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Warmup ratio: {warmup_ratio}")
    print(f"  Epochs: {num_train_epochs}")
    print(f"  Projection dim: 512 (2-layer MLP)")
    print(f"  LoRA rank: 128")
    print("="*60 + "\n")

    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        
        # GPU optimizations
        bf16=True if device == "cuda" else False,
        bf16_full_eval=True if device == "cuda" else False,
        dataloader_pin_memory=True,
        dataloader_num_workers=num_workers,
        dataloader_prefetch_factor=4,
        dataloader_persistent_workers=True,
        optim="adamw_torch_fused" if device == "cuda" else "adamw_torch",
        
        # Logging and saving
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=False,
        
        # Important flags
        remove_unused_columns=False,
        label_names=["labels"],
        
        # Learning rate schedule
        warmup_ratio=warmup_ratio,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        
        # Memory optimizations
        auto_find_batch_size=False,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=clip_data_collator,
        compute_loss_func=compute_clip_loss,
        callbacks=[SaveAdditionalWeightsCallback(output_dir)],
    )

    # Train
    trainer.train()

    # Save final model
    trainer.save_model(output_dir)
    
    if hasattr(model, 'module'):
        clip_model = model.module.model
    else:
        clip_model = model.model
    clip_model.save_pretrained(output_dir)
    
    print("\n" + "="*60)
    print(f"Training complete! Model saved to {output_dir}")
    print("="*60 + "\n")
    
    writer.close()

    return model, processor


def test(ckpt_path: str = "clip_model", val_dataset: str = "valid_grader", show_failures: bool = True, max_failures: int = 20):
    """Test CLIP model on multiple-choice QA dataset."""
    import tqdm

    testset = MultiChoiceQADataset(val_dataset)
    clip = load(ckpt_path)
    clip = clip.to(device)
    clip.eval()

    image_processor = tv.transforms.Compose([
        tv.transforms.Resize(256),
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

        # Add EOS token to candidates
        candidates_with_eos = [s + processor.tokenizer.eos_token for s in pair["candidates"]]
        
        text_inputs = processor(
            text=candidates_with_eos,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        )
        input_ids = text_inputs["input_ids"].long().to(device)
        attention_mask = text_inputs["attention_mask"].to(device)

        with torch.no_grad():
            _, _, logits = clip(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)

        prediction = logits.argmax(dim=-1).item()
        
        if prediction == pair["correct_index"]:
            correct_count += 1
        else:
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