from pathlib import Path
import json
from tqdm import tqdm

from .cot import CoTModel
from .data import Dataset


def _parse_answer_text(text: str):
    """Extract float value inside <answer>...</answer> tags."""
    import re

    match = re.search(r"<answer>(.*?)</answer>", text)
    if not match:
        return None
    try:
        return float(match.group(1).strip())
    except Exception:
        return None


def _is_close(a: float, b: float, rel_tol: float = 1e-3, abs_tol: float = 1e-3) -> bool:
    """RFT allows slight numeric differences."""
    try:
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
    except Exception:
        return False


def generate_dataset(output_json: str = "data/rft.json", oversample: int = 30, temperature: float = 0.8):
    """
    Generate an RFT dataset using rejection sampling.

    For each example in Dataset('train'):
      - use CoTModel.batched_generate() with num_return_sequences = oversample
      - pick the best completion whose <answer> matches ground truth
      - save [question, answer, reasoning_text] to output_json

    Args:
        output_json: path to save results (e.g., data/rft.json)
        oversample: number of completions to sample per question
        temperature: sampling temperature for diversity
    """
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ds = Dataset("train")
    
    # Use the larger 1.7B model for better quality reasoning chains
    print("Loading HuggingFaceTB/SmolLM2-1.7B-Instruct for high-quality CoT generation...")
    model = CoTModel(checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct")

    accepted = []

    for question, truth in tqdm(ds, desc="Generating RFT samples"):
        try:
            true_val = float(truth)
        except Exception:
            continue

        prompt = model.format_prompt(question)
        generations = model.batched_generate(
            [prompt],
            num_return_sequences=oversample,
            temperature=temperature,
        )

        # Normalize output from batched_generate (flatten if needed)
        if isinstance(generations, list) and len(generations) == 1 and isinstance(generations[0], list):
            generations = generations[0]

        # Find ALL correct completions, then pick the best one
        correct_completions = []
        for g in generations:
            pred_val = _parse_answer_text(g)
            if pred_val is not None and _is_close(pred_val, true_val):
                correct_completions.append(g)

        # Pick the completion with the most detailed reasoning (longest by word count)
        if correct_completions:
            chosen = max(correct_completions, key=lambda x: len(x.split()))
            accepted.append([question, true_val, chosen])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(accepted, f, indent=2, ensure_ascii=False)

    success_rate = len(accepted) / len(ds) * 100 if len(ds) > 0 else 0
    print(f"✅ Saved {len(accepted)}/{len(ds)} RFT examples to {output_path}")
    print(f"Success rate: {success_rate:.1f}%")
    return len(accepted)


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)