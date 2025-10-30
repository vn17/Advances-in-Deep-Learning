from .base_llm import BaseLLM
from .sft import format_example, tokenize as sft_tokenize
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
from pathlib import Path
import json


def load() -> BaseLLM:
    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


class RFTDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, rft_json_path: str):
        with open(rft_json_path, "r", encoding="utf-8") as f:
            entries = json.load(f)  # [question, numeric_answer, reasoning_text]

        # Each entry is a (prompt, answer) pair
        self.pairs = [(q, reasoning) for q, num, reasoning in entries]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        q, ans = self.pairs[idx]
        formatted = format_example(q, ans)
        return sft_tokenize(self.tokenizer, **formatted)


def collate_batch(batch):
    return {k: torch.tensor([d[k] for d in batch]) for k in batch[0]}


def train_model(
    output_dir: str,
    rft_json_path: str = "data/rft.json",
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 2e-5,
    **kwargs,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model and tokenizer
    llm = BaseLLM()
    model = llm.model
    tokenizer = llm.tokenizer
    device = llm.device

    # Load RFT dataset
    dataset = RFTDataset(tokenizer, rft_json_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

    # LoRA config
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
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
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

    model.save_pretrained(output_dir)
    print(f"Saved RFT LoRA adapter to {output_dir}")


if __name__ == "__main__":
    from fire import Fire
    Fire({"train": train_model, "load": load})