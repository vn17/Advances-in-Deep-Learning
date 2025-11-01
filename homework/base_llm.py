from typing import overload
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class BaseLLM:
    def __init__(self, checkpoint=checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.device = device
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token  # ensure pad_token is defined

    def format_prompt(self, question: str) -> str:
        return question

    def parse_answer(self, answer: str) -> float:
        print("\n-------ANS-------\n" + answer + "\n--------DONE------\n")
        try:
            return float(answer.split("<answer>")[1].split("</answer>")[0])
        except (IndexError, ValueError):
            return float("nan")

    def generate(self, prompt: str) -> str:
        return self.batched_generate([prompt])[0]

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: None = None, temperature: float = 0
    ) -> list[str]:
        ...

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: int, temperature: float = 0
    ) -> list[list[str]]:
        ...

    def batched_generate(
    self, prompts: list[str], num_return_sequences: int | None = None, temperature: float = 0
) -> list[str] | list[list[str]]:
        micro_batch_size = 32

        if len(prompts) > micro_batch_size:
            return [
                r
                for idx in tqdm(
                    range(0, len(prompts), micro_batch_size),
                    desc=f"LLM Running on Micro Batches {micro_batch_size}",
                )
                for r in self.batched_generate(
                    prompts[idx:idx + micro_batch_size], num_return_sequences, temperature
                )
            ]

        # --- Tokenize ---
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(self.device)
        input_length = inputs['input_ids'].shape[1]  # Store the input length

        # --- Generation settings ---
        do_sample = temperature > 0
        num_return_sequences = num_return_sequences or 1

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                num_return_sequences=num_return_sequences,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # --- Decode only the NEW tokens (exclude the prompt) ---
        # Slice from input_length onwards to get only generated tokens
        new_tokens = outputs[:, input_length:]
        decoded = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        # If multiple return sequences requested, reshape
        if num_return_sequences > 1:
            return [
                decoded[i * num_return_sequences:(i + 1) * num_return_sequences]
                for i in range(len(prompts))
            ]
        else:
            return decoded

    def answer(self, *questions) -> list[float]:
        prompts = [self.format_prompt(q) for q in questions]
        generations = self.batched_generate(prompts)
        return [self.parse_answer(g) for g in generations]


def test_model():
    testset = ["The cat went up", "The dog went down"]
    model = BaseLLM()
    for t in testset:
        print("testing generate function")
        print("input:", t)
        answer = model.generate(t)
        print("output:", answer)
    answers = model.batched_generate(testset)
    print("batched outputs:", answers)


if __name__ == "__main__":
    from fire import Fire
    Fire({"test": test_model})