from homework.base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Format prompt using Chain-of-Thought:
        - Give the model multiple example reasoning chains
        - Then ask the actual question with clear instructions
        - Always end with <answer>NUMBER</answer>
        """
        # Example reasoning questions and answers - focus on unit conversions
        examples = [
            ("Convert 2 kilograms to grams.",
             "1 kg = 1000 g, so 2 kg = 2 × 1000 = 2000 g.\n<answer>2000</answer>"),
            ("How many meters are in 3 kilometers?",
             "1 km = 1000 m, so 3 km = 3 × 1000 = 3000 m.\n<answer>3000</answer>"),
            ("Convert 5 hours to minutes.",
             "1 hour = 60 minutes, so 5 hours = 5 × 60 = 300 minutes.\n<answer>300</answer>"),
            ("How many milliliters are in 2 liters?",
             "1 L = 1000 mL, so 2 L = 2 × 1000 = 2000 mL.\n<answer>2000</answer>"),
        ]

        # Build messages list
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
                ),
            }
        ]

        # Add example Q/A pairs
        for q, a in examples:
            messages.append({"role": "user", "content": q})
            messages.append({"role": "assistant", "content": a})

        # Add the actual user question with simplified instruction
        messages.append({
            "role": "user",
            "content": f"{question}\n\nReason briefly, then end with <answer>...</answer>.",
        })

        # Apply chat template
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