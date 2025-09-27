import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from pathlib import Path
import math

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401
from .low_precision import Linear4Bit


class QLoRALinear(Linear4Bit):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_dim: int,
        group_size: int = 16,
        bias: bool = True,
    ) -> None:
        super().__init__(in_features, out_features, bias, group_size)
        self.requires_grad_(False)  # Freeze the base quantized weights

        # Initialize LoRA layers in float32 and move to DEVICE
        self.lora_a = torch.nn.Linear(in_features, lora_dim, bias=False, dtype=torch.float32).to(DEVICE)
        self.lora_b = torch.nn.Linear(lora_dim, out_features, bias=False, dtype=torch.float32).to(DEVICE)

        # Initialize LoRA weights with Kaiming + scale
        scale = 0.01
        torch.nn.init.kaiming_uniform_(self.lora_a.weight)
        self.lora_a.weight.data.mul_(scale)
        torch.nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base output using quantized weights
        base_out = super().forward(x)
        
        # LoRA path in float32
        x_32 = x.to(torch.float32).to(x.device)
        lora_out = self.lora_b(self.lora_a(x_32))
        
        # Combine outputs without scaling
        return base_out + lora_out.to(base_out.dtype).to(x.device)


class QLoRABigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels, lora_dim, group_size):
            super().__init__()
            self.model = torch.nn.Sequential(
                QLoRALinear(channels, channels, lora_dim, group_size).to(DEVICE),
                torch.nn.ReLU(),
                QLoRALinear(channels, channels, lora_dim, group_size).to(DEVICE),
                torch.nn.ReLU(),
                QLoRALinear(channels, channels, lora_dim, group_size).to(DEVICE),
            ).to(DEVICE)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Residual connection
            out = self.model(x.to(x.device))
            return out + x

    def __init__(self, lora_dim: int = 32, group_size: int = 128):
        super().__init__()
        # Define architecture to match BigNet exactly
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM, lora_dim, group_size).to(DEVICE),
            LayerNorm(BIGNET_DIM).to(DEVICE),
            self.Block(BIGNET_DIM, lora_dim, group_size).to(DEVICE),
            LayerNorm(BIGNET_DIM).to(DEVICE),
            self.Block(BIGNET_DIM, lora_dim, group_size).to(DEVICE),
            LayerNorm(BIGNET_DIM).to(DEVICE),
            self.Block(BIGNET_DIM, lora_dim, group_size).to(DEVICE),
            LayerNorm(BIGNET_DIM).to(DEVICE),
            self.Block(BIGNET_DIM, lora_dim, group_size).to(DEVICE),
            LayerNorm(BIGNET_DIM).to(DEVICE),
            self.Block(BIGNET_DIM, lora_dim, group_size).to(DEVICE),
        ).to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Move input to the same device as the model
        x = x.to(next(self.model.parameters()).device)
        return self.model(x)


def load(path: Path | None) -> QLoRABigNet:
    net = QLoRABigNet().to(DEVICE)
    if path is not None:
        net.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True), strict=False)
    return net
