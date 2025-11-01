from homework.base_llm import BaseLLM  # use absolute import to avoid relative import errors


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Format prompt using Chain-of-Thought:
        - Give the model an example reasoning chain
        - Then ask the actual question with clear instructions
        - Always end with <answer>NUMBER</answer>
        """
        # Example reasoning question and answer
        example_q = "If a box has 3 red balls and 2 blue balls, how many total balls are there?"
        example_a = "There are 3 red + 2 blue = 5 total balls.\n<answer>5</answer>"

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a concise reasoning assistant. "
                    "Show step-by-step reasoning briefly, then output only the numeric answer "
                    "in <answer> tags, e.g. <answer>42</answer>."
                ),
            },
            {"role": "user", "content": example_q},
            {"role": "assistant", "content": example_a},
            {
                "role": "user",
                "content": (
                    f"{question}\n\n"
                    "Reason step by step, then end your reply with the final numeric answer "
                    "inside <answer> tags. Do not include units or extra text."
                ),
            },
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from homework.data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire
    Fire({"test": test_model, "load": load})