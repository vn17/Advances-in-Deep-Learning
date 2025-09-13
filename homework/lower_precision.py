from pathlib import Path
import torch
import math
from .bignet import BIGNET_DIM, LayerNorm


def block_quantize_4bit(x: torch.Tensor, group_size: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize a 1D tensor to 4-bit per parameter using block scaling.
    Returns packed 4-bit weights and per-block scale factors.
    """
    assert x.dim() == 1
    x = x.view(-1, group_size)
    scale = x.abs().max(dim=-1, keepdim=True).values / 7  # map -7..7
    q = torch.clamp((x / scale).round(), -8, 7).to(torch.int8)
    # Pack 2 weights per byte
    packed = (q[:, ::2] & 0x0F) | ((q[:, 1::2] & 0x0F) << 4)
    return packed, scale

def block_dequantize_4bit(packed: torch.Tensor, scale: torch.Tensor, group_size: int = 8) -> torch.Tensor:
    """
    Dequantize 4-bit packed weights to float tensor
    """
    # Unpack 2 weights per byte
    q = torch.zeros(packed.size(0), group_size, device=packed.device, dtype=torch.float32)
    q[:, ::2] = (packed & 0x0F).float()
    q[:, 1::2] = ((packed >> 4) & 0x0F).float()
    q = q * scale
    return q.view(-1)

class LowerLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        # Register buffers for quantized weights
        shape = (out_features * in_features) // group_size
        self.register_buffer('weight_packed', torch.zeros(shape, dtype=torch.uint8))
        self.register_buffer('weight_scale', torch.zeros(shape, 1, dtype=torch.float32))

        # Learnable per-group bias (float32 for numerical stability)
        self.weight_bias = torch.nn.Parameter(torch.zeros(shape, 1, dtype=torch.float32))

        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize weights
        weight = block_dequantize_4bit(
            self.weight_packed, self.weight_scale,
            self.group_size
        )
        # Add learnable per-group bias before reshape
        weight = weight + self.weight_bias.view(-1)
        weight = weight.view(self.out_features, self.in_features)
        return torch.nn.functional.linear(x, weight, self.bias)

class LowerBigNet(torch.nn.Module):
    """
    BigNet implementation using mixed 2-bit/3-bit precision
    """
    class Block(torch.nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.model = torch.nn.Sequential(
                torch.nn.Linear(channels, channels),
                torch.nn.ReLU(),
                torch.nn.Linear(channels, channels),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # Apply residual scaling
            return 0.9 * self.model(x) + x

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def load(path: Path | None) -> LowerBigNet:
    net = LowerBigNet()
    if path is not None:
        state_dict = torch.load(path, weights_only=True)
        # Quantize the weights before loading
        new_state_dict = {}
        for name, param in state_dict.items():
            if 'weight' in name and 'norm' not in name:
                packed, scale = block_quantize_4bit(param.view(-1))
                new_state_dict[name + '_packed'] = packed
                new_state_dict[name + '_scale'] = scale
            else:
                new_state_dict[name] = param
        net.load_state_dict(new_state_dict, strict=False)
    return net
