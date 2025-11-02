import torch
from .base_llm import BaseLLM
from .data import Dataset, benchmark
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


class SFTModel(BaseLLM):
    """SFT Model with proper prompt formatting"""
    def format_prompt(self, question: str) -> str:
        """Format prompt to match the training format"""
        return (
            "system\n"
            "You are a helpful AI assistant named SmolLM, trained by Hugging Face\n"
            "user\n"
            f"{question}\n\nReason briefly, then end with <answer>...</answer>.\n"
            "assistant\n"
        )


def load() -> SFTModel:
    from pathlib import Path
    from peft import PeftModel

    model_name = "sft_model"
    model_path = Path(__file__).parent / model_name

    llm = SFTModel()  # Use SFTModel instead of BaseLLM
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def tokenize(tokenizer, question: str, answer: str):
    """Tokenize question and answer, masking the question in labels"""
    full_text = f"{question}{answer}{tokenizer.eos_token}"
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize the full text
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=256)
    
    # Tokenize just the question to find where to start labels
    question_tokens = tokenizer(question, add_special_tokens=False)
    question_len = len(question_tokens["input_ids"])
    
    # Create labels: mask question tokens with -100, keep answer tokens
    input_ids = full["input_ids"]
    labels = [-100] * question_len + input_ids[question_len:]
    
    # Mask padding tokens
    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100
    
    full["labels"] = labels
    return full


def format_example(prompt: str, answer: str) -> dict[str, str]:
    """Format training example using the model's chat template"""
    try:
        ans = round(float(answer), 3)
        answer_str = f"<answer>{ans}</answer>"
    except ValueError:
        answer_str = f"<answer>{answer}</answer>"

    # Use the same format that works in CoT
    question = (
        "system\n"
        "You are a helpful AI assistant named SmolLM, trained by Hugging Face\n"
        "user\n"
        f"{prompt}\n\nReason briefly, then end with <answer>...</answer>.\n"
        "assistant\n"
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


def collate_batch(batch):
    return {k: torch.tensor([d[k] for d in batch]) for k in batch[0]}


def train_model(output_dir: str = "sft_model", **kwargs):
    import torch
    from torch.utils.data import DataLoader
    from peft import LoraConfig, get_peft_model
    from tqdm import tqdm

    llm = BaseLLM()
    model = llm.model
    tokenizer = llm.tokenizer
    device = llm.device

    # LoRA configuration
    lora_config = LoraConfig(
        r=32,  # Increased rank for better capacity
        lora_alpha=64,  # Increased alpha (typically 2x rank)
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Target more layers
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config).to(device)
    model.train()
    
    print(f"Trainable parameters: {model.print_trainable_parameters()}")

    # Load and prepare data
    trainset = Dataset("train")
    tokenized_train = TokenizedDataset(tokenizer, trainset, format_example)
    dataloader = DataLoader(
        tokenized_train, 
        batch_size=4,  # Reduced batch size for stability
        shuffle=True, 
        collate_fn=collate_batch
    )

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    num_epochs = 5  # More epochs
    total_steps = len(dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(0.1 * total_steps), 
        num_training_steps=total_steps
    )

    # Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        
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
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}: avg_loss={avg_loss:.4f}")

    # Save model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved LoRA model to {output_dir}")
    
    # Test the model
    test_model(output_dir)


def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = SFTModel()  # Use SFTModel here too!
    from peft import PeftModel

    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)
    llm.model.eval()
    
    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire
    Fire({"train": train_model, "test": test_model, "load": load})