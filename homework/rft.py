from .base_llm import BaseLLM
from .sft import SFTModel, tokenize as sft_tokenize
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model, PeftModel
from pathlib import Path
import json
from tqdm import tqdm


class RFTModel(SFTModel):
    """RFT Model - inherits the same prompt formatting as SFT"""
    pass


def load() -> RFTModel:
    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = RFTModel()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


class RFTDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, rft_json_path: str):
        with open(rft_json_path, "r", encoding="utf-8") as f:
            entries = json.load(f)

        self.pairs = [(q, reasoning) for q, num, reasoning in entries]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        q, reasoning_answer = self.pairs[idx]
        
        question = (
            "system\n"
            "You are a helpful AI assistant named SmolLM, trained by Hugging Face\n"
            "user\n"
            f"{q}\n\nReason briefly, then end with <answer>...</answer>.\n"
            "assistant\n"
        )
        return sft_tokenize(self.tokenizer, question=question, answer=reasoning_answer)


def collate_batch(batch):
    return {k: torch.tensor([d[k] for d in batch]) for k in batch[0]}


def train_model(
    output_dir: str = "rft_model",
    rft_json_path: str = "data/rft.json",
    epochs: int = 5,  # More epochs
    batch_size: int = 2,  # Smaller batch for more updates
    lr: float = 3e-4,  # Lower LR for stability
    **kwargs,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    llm = BaseLLM()
    model = llm.model
    tokenizer = llm.tokenizer
    device = llm.device

    dataset = RFTDataset(tokenizer, rft_json_path)
    print(f"Loaded {len(dataset)} RFT training examples")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

    # Larger LoRA adapter (still under 50MB)
    lora_config = LoraConfig(
        r=64,  # Maximum capacity
        lora_alpha=128,  # 2x rank
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config).to(device)
    model.train()
    
    print(f"Trainable parameters: {model.print_trainable_parameters()}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps), 
        num_training_steps=total_steps
    )

    for epoch in range(epochs):
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs} avg_loss={avg:.4f}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved RFT LoRA adapter to {output_dir}")
    
    test_model(output_dir)


def test_model(ckpt_path: str):
    from .data import Dataset, benchmark
    
    testset = Dataset("valid")
    llm = RFTModel()
    
    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)
    llm.model.eval()
    
    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire
    Fire({"train": train_model, "test": test_model, "load": load})