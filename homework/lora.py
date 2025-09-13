from pathlib import Path

import math
import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401
from .half_precision import HalfLinear


class LoRALinear(HalfLinear):
    lora_a: torch.nn.Module
    lora_b: torch.nn.Module

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_dim: int,
        bias: bool = True,
    ) -> None:
        """
        Implement the LoRALinear layer as described in the homework

        Hint: You can use the HalfLinear class as a parent class (it makes load_state_dict easier, names match)
        Hint: Remember to initialize the weights of the lora layers
        Hint: Make sure the linear layers are not trainable, but the LoRA layers are
        """
        super().__init__(in_features, out_features, bias)

        # Initialize LoRA layers
        self.lora_a = torch.nn.Linear(in_features, lora_dim, bias=False, dtype=torch.float32)
        self.lora_b = torch.nn.Linear(lora_dim, out_features, bias=False, dtype=torch.float32)

        # Initialize LoRA weights
        torch.nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        torch.nn.init.zeros_(self.lora_b.weight)

        # Freeze the base HalfLinear weights
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cast input to float32 for LoRA layers
        x_fp32 = x.to(torch.float32)

        # Compute base HalfLinear output
        base_output = super().forward(x_fp32)

        # Compute LoRA output
        lora_output = self.lora_b(self.lora_a(x_fp32))

        # Combine base output and LoRA output, and cast back to input dtype
        return (base_output + lora_output).to(x.dtype)


class LoraBigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels: int, lora_dim: int):
            super().__init__()
            # Replace HalfLinear with LoRALinear and add LayerNorm
            self.model = torch.nn.Sequential(
                LoRALinear(channels, channels, lora_dim),
                LayerNorm(channels),  # LayerNorm should remain in full precision
                torch.nn.GELU(),
                LoRALinear(channels, channels, lora_dim),
            )

        def forward(self, x: torch.Tensor):
            # Residual connection
            return self.model(x) + x

    def __init__(self, lora_dim: int = 32):
        super().__init__()
        # Define the LoraBigNet architecture
        self.model = torch.nn.Sequential(
            LoraBigNet.Block(BIGNET_DIM, lora_dim),
            LoraBigNet.Block(BIGNET_DIM, lora_dim),
            LoraBigNet.Block(BIGNET_DIM, lora_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> LoraBigNet:
    # Since we have additional layers, we need to set strict=False in load_state_dict
    net = LoraBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net
