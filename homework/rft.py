from .base_llm import BaseLLM
from .sft import format_example, TokenizedDataset  # reuse SFT helpers
from .sft import TokenizedDataset as SFTTokenizedDataset  # alias if needed
from .sft import tokenize as sft_tokenize
from .sft import TokenizedDataset as _TD  # compatibility

def load() -> BaseLLM:
    from pathlib import Path
    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def train_model(
    output_dir: str,
    rft_json_path: str = "data/rft.json",
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 2e-5,
    device_batching: bool = True,
    **kwargs,
):
    """
    Train LoRA adapters on the RFT dataset (question + chain-of-thought reasoning).
    This reuses the SFT tokenization/formatting helpers (format_example, tokenize, TokenizedDataset).
    """

    import json
    import torch
    from pathlib import Path
    from torch.utils.data import DataLoader
    from transformers import AdamW, get_linear_schedule_with_warmup
    from peft import LoraConfig, get_peft_model

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load base model + tokenizer
    llm = BaseLLM()
    model = llm.model
    tokenizer = llm.tokenizer
    device = llm.device

    # Load RFT data
    with open(rft_json_path, "r", encoding="utf-8") as f:
        entries = json.load(f)  # each entry is [question, numeric_answer, reasoning_text]

    # Convert entries into (prompt, answer) pairs expected by format_example
    # format_example expects (prompt, answer_str) where answer_str becomes <answer>..</answer>
    # But our entries already have reasoning_text that should include <answer> tags. We'll pass
    # reasoning_text as `answer` and prompt as `question`.
    qa_pairs = [(q, reasoning) for q, num, reasoning in entries]

    # Build dataset
    class SimpleIterableDataset(torch.utils.data.Dataset):
        def __init__(self, pairs, tokenizer, format_fn):
            self.pairs = pairs
            self.tokenizer = tokenizer
            self.format_fn = format_fn

        def __len__(self):
            return len(self.pairs)

        def __getitem__(self, idx):
            q, ans = self.pairs[idx]
            formatted = format_example(q, ans)
            # reuse tokenize from sft module if available; otherwise implement the same logic here
            # Import tokenize from sft to ensure identical behaviour.
            from .sft import tokenize as tokenize_fn
            return tokenize_fn(tokenizer, **formatted)

    dataset = SimpleIterableDataset(qa_pairs, tokenizer, format_example)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: {
        k: torch.tensor([d[k] for d in x]) for k in x[0]
    })

    # LoRA config - slightly larger rank than SFT to capture reasoning patterns
    lora_config = LoraConfig(
        r=16,  # increase rank a bit
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
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

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

    # Save the LoRA adapter
    model.save_pretrained(output_dir)
    print(f"Saved RFT LoRA adapter to {output_dir}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "load": load})