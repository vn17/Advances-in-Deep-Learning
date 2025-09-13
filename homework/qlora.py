from pathlib import Path
import math

import torch

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

        # Initialize LoRA layers
        self.lora_a = torch.nn.Linear(in_features, lora_dim, bias=False, dtype=torch.float32)
        self.lora_b = torch.nn.Linear(lora_dim, out_features, bias=False, dtype=torch.float32)

        # Initialize LoRA weights
        torch.nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_b.weight)

        # Ensure LoRA layers are trainable
        self.lora_a.requires_grad_(True)
        self.lora_b.requires_grad_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast input to float32 for LoRA layers
        x_fp32 = x.to(torch.float32)

        # Compute base Linear4Bit output
        base_output = super().forward(x_fp32)

        # Compute LoRA output
        lora_output = self.lora_b(self.lora_a(x_fp32))

        # Combine base output and LoRA output, and cast back to input dtype
        return (base_output + lora_output).to(x.dtype)


class QLoRABigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels, lora_dim, group_size):
            super().__init__()
            # Replace Linear4Bit with QLoRALinear and add LayerNorm
            self.model = torch.nn.Sequential(
                QLoRALinear(channels, channels, lora_dim, group_size),
                LayerNorm(channels),  # LayerNorm should remain in full precision
                torch.nn.GELU(),
                QLoRALinear(channels, channels, lora_dim, group_size),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Residual connection
            return self.model(x) + x

    def __init__(self, lora_dim: int = 32, group_size: int = 16):
        super().__init__()
        # Define the QLoRABigNet architecture
        self.model = torch.nn.Sequential(
            QLoRABigNet.Block(BIGNET_DIM, lora_dim, group_size),
            QLoRABigNet.Block(BIGNET_DIM, lora_dim, group_size),
            QLoRABigNet.Block(BIGNET_DIM, lora_dim, group_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> QLoRABigNet:
    net = QLoRABigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net
