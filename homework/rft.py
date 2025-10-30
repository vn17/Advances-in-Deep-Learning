from .base_llm import BaseLLM
from .sft import format_example, tokenize as sft_tokenize
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model


def load() -> BaseLLM:
    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


class RFTDataset(torch.utils.data.Dataset):
    def __init__(self, entries, tokenizer):
        """
        entries: list of [question:str, numeric_answer:float, reasoning_text:str]
        """
        self.entries = entries
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        q, num, reasoning = self.entries[idx]
        # reasoning_text already contains <answer> tags
        formatted = format_example(q, reasoning)
        return sft_tokenize(self.tokenizer, **formatted)


def collate_fn(batch):
    return {k: torch.tensor([d[k] for d in batch]) for k in batch[0]}


def train_model(
    output_dir: str,
    rft_json_path: str = "data/rft.json",
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 2e-5,
    **kwargs,
):
    """
    Train LoRA adapters on the RFT dataset (question + chain-of-thought reasoning).
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base model + tokenizer
    llm = BaseLLM()
    model = llm.model
    tokenizer = llm.tokenizer
    device = llm.device

    # Load RFT data
    with open(rft_json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)

    dataset = RFTDataset(entries, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # LoRA config (slightly larger than SFT)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config).to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(dataloader) * epochs if len(dataloader) > 0 else 1
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        avg = total_loss / (len(dataloader) if len(dataloader) > 0 else 1)
        print(f"Epoch {epoch + 1}/{epochs} avg_loss={avg:.4f}")

    # Save LoRA adapter
    model.save_pretrained(output_dir)
    print(f"Saved RFT LoRA adapter to {output_dir}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "load": load})