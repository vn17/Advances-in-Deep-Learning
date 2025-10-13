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
    Input:  (B, h, w) integer tokens
    Output: (B, h, w, n_tokens) probability logits over next tokens.
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10, n_heads: int = 8, n_layers: int = 6):
        super().__init__()
        self.n_tokens = n_tokens
        self.d_latent = d_latent

        # Token embedding
        self.token_emb = nn.Embedding(n_tokens, d_latent)

        # Optional positional embedding (learned)
        self.pos_emb = nn.Embedding(1024, d_latent)  # supports sequences up to 1024

        # Transformer stack (encoder used as causal decoder)
        layer = nn.TransformerEncoderLayer(
            d_model=d_latent,
            nhead=n_heads,
            dim_feedforward=d_latent * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,  # easier to reason about (B, seq, d)
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

        # Output projection back to token logits
        self.output = nn.Linear(d_latent, n_tokens)

    # -----------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
      """
      x: LongTensor (B, h, w) or (B, seq)
      Returns:
          logits: (B, h, w, n_tokens)
          metrics: (empty for now)
      """
      if x.dim() == 3:
          B, h, w = x.shape
          seq_len = h * w
          x = x.view(B, seq_len)
      else:
          B, seq_len = x.shape
          # try to infer h, w from sequence length (square or rectangular guess)
          h = int(math.sqrt(seq_len))
          w = seq_len // h

      device = x.device

      # Shifted input: prepend a BOS (all zeros) token, remove last element
      bos = torch.zeros((B, 1), device=device, dtype=x.dtype)
      x_shifted = torch.cat([bos, x[:, :-1]], dim=1)

      # Token + positional embedding
      token_emb = self.token_emb(x_shifted)  # (B, seq, d)
      pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
      pos_emb = self.pos_emb(pos_ids)
      emb = token_emb + pos_emb  # (B, seq, d)

      # Create causal mask (True = mask)
      mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()

      # Forward through transformer
      out = self.transformer(emb, mask=mask)  # (B, seq, d)

      # Project to vocab logits
      logits = self.output(out)  # (B, seq, n_tokens)

      # Reshape back into image grid
      logits = logits.view(B, h, w, self.n_tokens)

      metrics = {}
      return logits, metrics

    # -----------------------------------------------------------------
    # Generation
    # -----------------------------------------------------------------
    @torch.no_grad()
    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:
        """
        Generate autoregressively one token at a time.
        """
        device = device or next(self.parameters()).device
        seq_len = h * w

        x = torch.zeros((B, seq_len), dtype=torch.long, device=device)

        for i in range(seq_len):
            logits, _ = self.forward(x)
            # logits: (B, h, w, n_tokens) -> flatten sequence
            logits_seq = logits.view(B, seq_len, self.n_tokens)
            probs = F.softmax(logits_seq[:, i, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x[:, i] = next_token.squeeze(-1)

        return x.view(B, h, w)