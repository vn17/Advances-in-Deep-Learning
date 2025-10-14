from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image

from .autoregressive import Autoregressive
from .bsq import Tokenizer


class Compressor:
    def __init__(self, tokenizer: Tokenizer, autoregressive: Autoregressive):
        self.tokenizer = tokenizer
        self.autoregressive = autoregressive
        self.device = next(self.autoregressive.parameters()).device

    # ------------------ Compression helpers ------------------
    def _encode_arithmetic(self, token, logits):
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        cdf = np.cumsum(probs)
        cdf = np.insert(cdf, 0, 0.0)
        return cdf[token], cdf[token + 1]

    def _decode_arithmetic(self, code, logits):
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        cdf = np.cumsum(probs)
        cdf = np.insert(cdf, 0, 0.0)
        idx = np.searchsorted(cdf, code) - 1
        code = (code - cdf[idx]) / (cdf[idx + 1] - cdf[idx] + 1e-12)
        return idx, code

    # ------------------ Main API ------------------
    def compress(self, x: torch.Tensor) -> bytes:
        self.tokenizer.eval()
        self.autoregressive.eval()

        # Add batch dimension if needed
        if x.ndim == 3:
            x = x.unsqueeze(0)  # (1, H, W, 3)
        
        tokens_2d = self.tokenizer.encode_index(x)  # (B, h, w)
        tokens_2d = tokens_2d.squeeze(0)  # Remove batch dim -> (h, w)
        h, w = tokens_2d.shape
        tokens = tokens_2d.flatten()  # 1D sequence for arithmetic coding

        low, high = 0.0, 1.0

        # Store h and w dimensions as metadata (prepend to compressed bytes)
        metadata = np.array([h, w], dtype=np.int32)
        
        with torch.no_grad():
            for i in range(len(tokens)):
                # Create a 2D grid with padding for context
                if i > 0:
                    context_tokens = tokens[:i]
                    # Pad to full grid size
                    padded = torch.zeros(h * w, dtype=torch.long, device=self.device)
                    padded[:i] = context_tokens.to(self.device)
                    context_grid = padded.view(1, h, w)
                else:
                    context_grid = torch.zeros((1, h, w), dtype=torch.long, device=self.device)
                
                logits_grid, _ = self.autoregressive(context_grid)
                # Get logits for the current position
                row, col = i // w, i % w
                logits = logits_grid[0, row, col]
                
                l, h_ = self._encode_arithmetic(tokens[i].item(), logits)
                range_ = high - low
                high = low + range_ * h_
                low = low + range_ * l

        code = np.array([(low + high) / 2], dtype=np.float64)
        return metadata.tobytes() + code.tobytes()

    def decompress(self, x: bytes) -> torch.Tensor:
        self.tokenizer.eval()
        self.autoregressive.eval()

        # Extract metadata (h, w) and code
        metadata = np.frombuffer(x[:8], dtype=np.int32)  # First 8 bytes for two int32s
        h, w = int(metadata[0]), int(metadata[1])
        code = np.frombuffer(x[8:], dtype=np.float64)[0]  # Rest is the float64 code
        
        seq_len = h * w
        tokens = []

        with torch.no_grad():
            for i in range(seq_len):
                # Create a 2D grid with padding for context
                if i > 0:
                    padded = torch.zeros(seq_len, dtype=torch.long, device=self.device)
                    padded[:i] = torch.tensor(tokens, dtype=torch.long, device=self.device)
                    context_grid = padded.view(1, h, w)
                else:
                    context_grid = torch.zeros((1, h, w), dtype=torch.long, device=self.device)
                
                logits_grid, _ = self.autoregressive(context_grid)
                row, col = i // w, i % w
                logits = logits_grid[0, row, col]
                
                idx, code = self._decode_arithmetic(code, logits)
                tokens.append(idx)

        tokens = torch.tensor(tokens, device=self.device).view(1, h, w)
        img = self.tokenizer.decode_index(tokens)  # (1, H, W, 3)
        return img.squeeze(0).clamp(-0.5, 0.5)  # Remove batch dim -> (H, W, 3)


# ------------------ CLI API ------------------
def compress(tokenizer: Path, autoregressive: Path, image: Path, compressed_image: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    x = torch.tensor(np.array(Image.open(image)), dtype=torch.uint8, device=device)
    cmp_bytes = cmp.compress(x.float() / 255.0 - 0.5)
    with open(compressed_image, "wb") as f:
        f.write(cmp_bytes)


def decompress(tokenizer: Path, autoregressive: Path, compressed_image: Path, image: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tk_model = cast(Tokenizer, torch.load(tokenizer, weights_only=False).to(device))
    ar_model = cast(Autoregressive, torch.load(autoregressive, weights_only=False).to(device))
    cmp = Compressor(tk_model, ar_model)

    with open(compressed_image, "rb") as f:
        cmp_bytes = f.read()

    x = cmp.decompress(cmp_bytes)
    img = Image.fromarray(((x + 0.5) * 255.0).clamp(min=0, max=255).byte().cpu().numpy())
    img.save(image)


if __name__ == "__main__":
    from fire import Fire
    Fire({"compress": compress, "decompress": decompress})