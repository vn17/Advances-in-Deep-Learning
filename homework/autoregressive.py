import abc
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


# ---------------------------------------------------------------------
# Base Class
# ---------------------------------------------------------------------
class Autoregressive(abc.ABC):
    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        pass

    @abc.abstractmethod
    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:
        pass


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
class AutoregressiveModel(nn.Module, Autoregressive):
    """
    Auto-regressive model using TransformerEncoderLayer (causal transformer).
    Accepts inputs shaped (B, h, w), (B, seq_len), or (B, C, H, W).
    Outputs logits shaped (B, h, w, n_tokens).
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10, n_heads: int = 8, n_layers: int = 6):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_latent = d_latent

        # Token embedding
        self.token_emb = nn.Embedding(n_tokens, d_latent)

        # Positional embedding
        self.pos_emb = nn.Embedding(2048, d_latent)

        # Transformer
        layer = nn.TransformerEncoderLayer(
            d_model=d_latent,
            nhead=n_heads,
            dim_feedforward=d_latent * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

        # Output projection
        self.output = nn.Linear(d_latent, n_tokens)

    # -----------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Handles (B, h, w), (B, seq), or (B, C, H, W) gracefully.
        Returns logits: (B, h, w, n_tokens)
        """
        # 🔹 Step 1: Normalize input shape
        if x.dim() == 4:
            # If input is (B, C, H, W) → reduce to (B, H, W)
            if x.size(1) == 1:  # single channel tokens
                x = x.squeeze(1)
            else:
                raise ValueError(f"Unexpected 4D input with C={x.size(1)}")
        if x.dim() == 3:
            B, h, w = x.shape
            seq_len = h * w
            x = x.view(B, seq_len)
        elif x.dim() == 2:
            B, seq_len = x.shape
            h = int(math.sqrt(seq_len))
            w = seq_len // h
        else:
            raise ValueError(f"Unsupported input shape {x.shape}")

        device = x.device

        # 🔹 Step 2: Shift sequence for causal prediction
        bos = torch.zeros((B, 1), device=device, dtype=x.dtype)
        x_shifted = torch.cat([bos, x[:, :-1]], dim=1)

        # 🔹 Step 3: Embeddings
        token_emb = self.token_emb(x_shifted)
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_emb = self.pos_emb(pos_ids)
        emb = token_emb + pos_emb

        # 🔹 Step 4: Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

        # 🔹 Step 5: Transformer forward
        out = self.transformer(emb, mask=mask)

        # 🔹 Step 6: Project to logits
        logits = self.output(out)  # (B, seq, n_tokens)
        logits = logits.view(B, h, w, self.n_tokens)

        return logits, {}

    # -----------------------------------------------------------------
    @torch.no_grad()
    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:
        device = device or next(self.parameters()).device
        seq_len = h * w
        x = torch.zeros((B, seq_len), dtype=torch.long, device=device)

        for i in range(seq_len):
            logits, _ = self.forward(x)
            logits_seq = logits.view(B, seq_len, self.n_tokens)
            probs = F.softmax(logits_seq[:, i, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x[:, i] = next_token.squeeze(-1)

        return x.view(B, h, w)