from .base_llm import BaseLLM
from .sft import SFTModel, format_example, tokenize as sft_tokenize
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

    llm = RFTModel()  # Use RFTModel instead of BaseLLM
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


class RFTDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, rft_json_path: str):
        with open(rft_json_path, "r", encoding="utf-8") as f:
            entries = json.load(f)  # [question, numeric_answer, reasoning_text]

        # Each entry is a (prompt, answer) pair - use the reasoning_text which includes <answer> tags
        self.pairs = [(q, reasoning) for q, num, reasoning in entries]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        q, ans = self.pairs[idx]
        # Don't use format_example since reasoning already has the answer format
        # Just format the question part
        question = (
            "system\n"
            "You are a helpful AI assistant named SmolLM, trained by Hugging Face\n"
            "user\n"
            f"{q}\n\nReason briefly, then end with <answer>...</answer>.\n"
            "assistant\n"
        )
        return sft_tokenize(self.tokenizer, question=question, answer=ans)


def collate_batch(batch):
    return {k: torch.tensor([d[k] for d in batch]) for k in batch[0]}


def train_model(
    output_dir: str = "rft_model",
    rft_json_path: str = "data/rft.json",
    epochs: int = 5,
    batch_size: int = 4,
    lr: float = 1e-4,
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

    # LoRA config - match or exceed SFT configuration
    lora_config = LoraConfig(
        r=32,  # Match your successful SFT config
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Target more modules
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config).to(device)
    model.train()
    
    print(f"Trainable parameters: {model.print_trainable_parameters()}")

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(dataloader) * epochs if len(dataloader) > 0 else 1
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    for epoch in range(epochs):
        total_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg = total_loss / (len(dataloader) if len(dataloader) > 0 else 1)
        print(f"Epoch {epoch + 1}/{epochs} avg_loss={avg:.4f}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved RFT LoRA adapter to {output_dir}")
    
    # Test the model
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