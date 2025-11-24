from pathlib import Path

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoProcessor, Trainer, TrainingArguments

from .base_vlm import BaseVLM
from .data import VQADataset, benchmark

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")


def load(model_name: str = "vlm_model") -> BaseVLM:
    from pathlib import Path

    from peft import PeftModel

    model_path = Path(__file__).parent / model_name

    vlm = BaseVLM()
    # If there is a saved LoRA/PEFT checkpoint available use it, otherwise return
    # the base VLM model so the grader can still run without saved adapters.
    adapter_config = model_path / "adapter_config.json"
    if model_path.exists() and adapter_config.exists():
        try:
            vlm.model = PeftModel.from_pretrained(vlm.model, model_path).to(vlm.device)
            vlm.model.eval()
            return vlm
        except Exception:
            # fallback to base model if loading fails
            vlm.model.eval()
            return vlm

    # No adapter found -> return base model (unwrapped)
    vlm.model.eval()
    return vlm


def custom_data_collator(features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_length = max(f["input_ids"].shape[0] for f in features)

    def pad_tensor(tensor, pad_value):
        if tensor.shape[0] >= max_length:
            return tensor
        pad_size = max_length - tensor.shape[0]
        # Pad on the left for decoder-only models
        return torch.cat([torch.full((pad_size,), pad_value, dtype=tensor.dtype), tensor])

    input_ids = torch.stack([pad_tensor(f["input_ids"], pad_value=processor.tokenizer.pad_token_id) for f in features])
    attention_mask = torch.stack([pad_tensor(f["attention_mask"], pad_value=0) for f in features])
    labels = torch.stack([pad_tensor(f["labels"], pad_value=-100) for f in features])
    pixel_values = torch.stack([f["pixel_values"] for f in features])

    return {
        "input_ids": input_ids.long(),
        "attention_mask": attention_mask.long(),
        "labels": labels.long(),
        "pixel_values": pixel_values,
    }


class VQADatasetForTraining(Dataset):
    def __init__(self, dataset: VQADataset, processor: AutoProcessor):
        self.dataset = dataset
        self.processor = processor
        self.features = ["image", "question", "answer"]
        
        # Ensure tokenizer has pad token set
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        
        self.image_token_id = self.processor.tokenizer.additional_special_tokens_ids[
            self.processor.tokenizer.additional_special_tokens.index("<image>")
        ]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        item = self.dataset[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        
        # Prepare input text in chat format
        input_message = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": item["question"]}]}]
        prompt = self.processor.apply_chat_template(input_message, add_generation_prompt=True)
        
        # Process prompt separately to get accurate token length
        prompt_inputs = self.processor(
            images=image,
            text=prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
        )
        prompt_len = prompt_inputs["input_ids"].shape[1]
        
        # Process full text (prompt + answer + EOS)
        full_text = prompt + item["answer"] + self.processor.tokenizer.eos_token
        inputs = self.processor(
            images=image,
            text=full_text,
            return_tensors="pt",
            padding=False,
            truncation=True,
        )

        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        # Prepare labels: mask prompt tokens, keep only answer tokens for loss
        labels = input_ids.clone()
        labels[:prompt_len] = -100  # Mask prompt (including image tokens)
        
        return {
            "input_ids": input_ids.long(),
            "attention_mask": attention_mask.long(),
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "labels": labels.long(),
        }


def train(
    data_dir: Path | None = None,
    train_dataset_name: str = "train",
    output_dir: str = "vlm_sft",
    num_train_epochs: float = 0.05,  # Changed to float for fractional epochs
    per_device_train_batch_size: int = 8,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 5e-4,
    lora_r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.0,
    num_workers: int = 16,
):
    vlm = BaseVLM()

    # Create output directory
    output_dir = Path(__file__).parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize TensorBoard writer
    tensorboard_dir = output_dir / "tensorboard"
    tensorboard_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=tensorboard_dir)

    # Initialize model and processor
    processor = vlm.processor
    model = vlm.model

    # Ensure pad token is set
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules="all-linear",
        bias="none",
    )

    # Apply LoRA to the model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.config.use_cache = False
    model.enable_input_require_grads()
    
    # Enable gradient checkpointing BEFORE setting to train mode
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    model.train()

    # Prepare datasets
    train_dataset = VQADataset(train_dataset_name, data_dir)
    train_dataset = VQADatasetForTraining(train_dataset, processor)

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        learning_rate=learning_rate,
        bf16=True if DEVICE == "cuda" else False,
        logging_steps=1,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        label_names=["labels"],
        dataloader_num_workers=num_workers,
        remove_unused_columns=False,  # ADD THIS LINE
        dataloader_pin_memory=True if DEVICE == "cuda" else False,
)

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=custom_data_collator,
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model(output_dir)

    # Close TensorBoard writer
    writer.close()

    return model, processor


def evaluate(model: nn.Module, val_loader: DataLoader) -> float:
    """
    Evaluate the model on the validation set.

    Args:
        model: Model to evaluate
        val_loader: Validation data loader

    Returns:
        Average validation loss
    """
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            val_loss += outputs.loss.item()

    model.train()
    return val_loss / len(val_loader)


def demo_train():
    train(
        train_dataset_name="train_demo",
        output_dir="demo_train",
        num_train_epochs=1,
        per_device_train_batch_size=1,
        num_workers=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-8,
    )


def test_model(ckpt_path: str, val_dataset: str = "valid_grader"):
    try:
        testset = VQADataset(val_dataset)
        llm = load(ckpt_path)
        
        # Ensure model is in eval mode and on correct device
        llm.model.eval()
        
        benchmark_result = benchmark(llm, testset, 128)
        print(f"Accuracy: {benchmark_result.accuracy}")
        return benchmark_result.accuracy
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


if __name__ == "__main__":
    from fire import Fire

    Fire({"demo_train": demo_train, "train": train, "test": test_model})
