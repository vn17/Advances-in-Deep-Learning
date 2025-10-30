from .base_llm import BaseLLM
from .data import Dataset, benchmark


def load() -> BaseLLM:
    from pathlib import Path
    from peft import PeftModel

    model_name = "sft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def tokenize(tokenizer, question: str, answer: str):
    """
    Tokenize a question/answer pair and produce labels.
    Only the answer portion is supervised (question tokens -> label = -100).
    """
    full_text = f"{question} {answer}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    input_ids = full["input_ids"]
    question_len = len(tokenizer(question)["input_ids"])

    # Mask prompt part (no loss)
    labels = [-100] * question_len + input_ids[question_len:]

    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full


def format_example(prompt: str, answer: str) -> dict[str, str]:
    """
    Construct a formatted prompt/answer pair suitable for fine-tuning.
    Adds reasoning guidance and <answer></answer> tags to standardize output.
    """
    # Round numeric answers for stability
    try:
        ans = round(float(answer), 3)
        answer_str = f"<answer>{ans}</answer>"
    except ValueError:
        answer_str = f"<answer>{answer}</answer>"

    question = (
        "You are a helpful reasoning assistant.\n"
        f"Question: {prompt}\n"
        "Think step by step and provide the numeric answer inside <answer> tags."
    )

    return {"question": question, "answer": answer_str}


class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formatted = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formatted)


def train_model(output_dir: str, **kwargs):
    """
    Fine-tune the base LLM using LoRA with PEFT.
    """
    import torch
    from torch.utils.data import DataLoader
    from peft import LoraConfig, get_peft_model
    from transformers import AdamW, get_linear_schedule_with_warmup

    # Load base model
    llm = BaseLLM()
    model = llm.model
    tokenizer = llm.tokenizer
    device = llm.device

    # LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],  # works well for transformer-based LLMs
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config).to(device)
    model.train()

    # Dataset
    trainset = Dataset("train")
    tokenized_train = TokenizedDataset(tokenizer, trainset, format_example)
    dataloader = DataLoader(tokenized_train, batch_size=8, shuffle=True)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(dataloader) * 3  # 3 epochs default
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0.1 * total_steps, num_training_steps=total_steps
    )

    for epoch in range(3):
        total_loss = 0
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}: avg_loss={total_loss / len(dataloader):.4f}")

    # Save the LoRA adapter
    model.save_pretrained(output_dir)
    print(f"Saved LoRA model to {output_dir}")

    test_model(output_dir)


def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = BaseLLM()

    from peft import PeftModel

    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})