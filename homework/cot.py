from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Convert a raw question into a chat-style prompt suitable for an
        instruction-tuned LLM (e.g., SmolLM2-Instruct).
        We explicitly ask the model to reason step-by-step and provide
        the final numeric answer enclosed in <answer></answer> tags.
        """

        # Construct a chat conversation list as expected by apply_chat_template
        messages = [
            {"role": "system", "content": "You are a helpful assistant that reasons step by step."},
            {
                "role": "user",
                "content": (
                    f"{question}\n\n"
                    "Think carefully step by step. "
                    "At the end, output your final numeric answer between <answer> and </answer> tags."
                ),
            },
        ]

        # Convert the messages into a single prompt string using the model's chat template
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})