import abc
import torch
import torch.nn as nn
from .ae import PatchAutoEncoder, hwc_to_chw, chw_to_hwc

# ------------------------------------------------------
# Differentiable Sign (Straight-through Estimator)
# ------------------------------------------------------
def diff_sign(x: torch.Tensor) -> torch.Tensor:
    sign = 2 * (x >= 0).float() - 1
    return x + (sign - x).detach()


# ------------------------------------------------------
# Base Tokenizer
# ------------------------------------------------------
class Tokenizer(abc.ABC):
    @abc.abstractmethod
    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        pass


# ------------------------------------------------------
# Binary Spherical Quantizer (BSQ)
# ------------------------------------------------------
class BSQ(nn.Module):
    def __init__(self, codebook_bits: int, embedding_dim: int):
        super().__init__()
        self._codebook_bits = codebook_bits
        self.down = nn.Linear(embedding_dim, codebook_bits)
        self.up = nn.Linear(codebook_bits, embedding_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten last dimension and project
        z = self.down(x)
        # L2 normalize
        z = z / (z.norm(dim=-1, keepdim=True) + 1e-8)
        # Binarize to ±1
        return diff_sign(z)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    # --------------------------------------------------
    # Code ↔ Index conversions
    # --------------------------------------------------
    def _code_to_index(self, x: torch.Tensor) -> torch.Tensor:
        x = (x >= 0).long()
        bits = 2 ** torch.arange(self._codebook_bits, device=x.device, dtype=torch.long)
        return (x * bits).sum(dim=-1)

    def _index_to_code(self, x: torch.Tensor) -> torch.Tensor:
        bits = torch.arange(self._codebook_bits, device=x.device)
        return 2 * ((x[..., None] & (1 << bits)) > 0).float() - 1


# ------------------------------------------------------
# BSQ Patch AutoEncoder
# ------------------------------------------------------
class BSQPatchAutoEncoder(PatchAutoEncoder, Tokenizer):
    def __init__(self, patch_size: int = 5, latent_dim: int = 128, codebook_bits: int = 10):
        super().__init__(patch_size=patch_size, latent_dim=latent_dim, bottleneck=latent_dim)
        self.codebook_bits = codebook_bits
        self.bsq = BSQ(codebook_bits, latent_dim)

    # ---------------- Encode/Decode -------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z = super().encode(x)          # (B, h, w, latent_dim)
        z = self.bsq.encode(z)         # (B, h, w, codebook_bits)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z_up = self.bsq.decode(z)      # (B, h, w, latent_dim)
        x_rec = super().decode(z_up)   # (B, H, W, 3)
        return x_rec

    # ---------------- Tokenizer API -------------------
    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        z = super().encode(x)          # (B, h, w, latent_dim)
        z = self.bsq.encode(z)         # (B, h, w, codebook_bits)
        return self.bsq._code_to_index(z)

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        z_bin = self.bsq._index_to_code(x)
        return self.decode(z_bin)

    # ---------------- Forward -------------------
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z = self.encode(x)
        x_rec = self.decode(z)

        # Compute token histogram safely
        tokens = self.bsq._code_to_index(z).reshape(-1)
        tokens = tokens.clamp(min=0)  # ensure non-negative
        cnt = torch.bincount(tokens, minlength=2 ** self.codebook_bits)

        metrics = {
            "cb0": (cnt == 0).float().mean().detach(),
            "cb2": (cnt <= 2).float().mean().detach(),
        }
        return x_rec, metrics


# ------------------------------------------------------
# Model Loader
# ------------------------------------------------------
def load() -> torch.nn.Module:
    from pathlib import Path
    model_name = "BSQPatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)