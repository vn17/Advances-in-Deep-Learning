from homework.base_llm import BaseLLM  # use absolute import to avoid relative import errors


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
      """
      Format prompt using Chain-of-Thought:
      - Give the model multiple example reasoning chains
      - Then ask the actual question with clear instructions
      - Always end with <answer>NUMBER</answer>
      """
      # Example reasoning questions and answers
      examples = [
          ("If a box has 3 red balls and 2 blue balls, how many total balls are there?",
          "There are 3 red + 2 blue = 5 total balls.\n<answer>5</answer>"),
          ("What is the sum of 7 and 5?",
          "Add 7 + 5 = 12.\n<answer>12</answer>"),
          ("If you have 10 apples and eat 4, how many are left?",
          "10 apples - 4 eaten = 6 remaining apples.\n<answer>6</answer>"),

          # Additional examples
          ("Convert 2 kilograms to grams.",
          "2 kg = 2 * 1000 g = 2000 g.\n<answer>2000</answer>"),
          # ("If a car travels 60 miles in 2 hours, what is its speed in mph?",
          # "Speed = distance / time = 60 / 2 = 30 mph.\n<answer>30</answer>"),
          # ("How many minutes are in 3 hours?",
          # "3 hours * 60 minutes/hour = 180 minutes.\n<answer>180</answer>"),
          # ("If a rectangle has length 5 m and width 3 m, what is its area?",
          # "Area = length * width = 5 * 3 = 15.\n<answer>15</answer>"),
          # ("Convert 4 liters to milliliters.",
          # "4 L * 1000 mL/L = 4000 mL.\n<answer>4000</answer>"),
          # ("A bag contains 12 candies, and you take 3, how many remain?",
          # "12 - 3 = 9 candies remaining.\n<answer>9</answer>"),
          # ("If a book costs $15 and you buy 3, what is the total cost?",
          # "Total cost = 15 * 3 = 45.\n<answer>45</answer>")
      ]

      # Build messages list
      messages = [
          {
              "role": "system",
              "content": (
                  "You are a concise reasoning assistant. "
                  "Show step-by-step reasoning briefly, then output only the numeric answer "
                  "in <answer> tags, e.g. <answer>42</answer>."
              ),
          }
      ]

      # Add example Q/A pairs
      for q, a in examples:
          messages.append({"role": "user", "content": q})
          messages.append({"role": "assistant", "content": a})

      # Add the actual user question
      messages.append({
          "role": "user",
          "content": (
              f"{question}\n\n"
              "Reason step by step, then end your reply with the final numeric answer "
              "inside <answer> tags. Do not include units or extra text."
          ),
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