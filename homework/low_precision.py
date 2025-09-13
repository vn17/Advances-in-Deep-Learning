from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401


def block_quantize_4bit(x: torch.Tensor, group_size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize the input tensor to 4-bit precision along the last dimension.
    Always quantize group_size value together and store their absolute value first.
    To keep things simple, we require x to be a 1D tensor, and the size divisible by group_size.
    Return the quantized tensor and scaling factor.
    """
    assert x.dim() == 1
    assert x.size(0) % group_size == 0

    x = x.view(-1, group_size)
    normalization = x.abs().max(dim=-1, keepdim=True).values
    x_norm = (x + normalization) / (2 * normalization)
    x_quant_8 = (x_norm * 15).round().to(torch.int8)
    x_quant_4 = (x_quant_8[:, ::2] & 0xF) + ((x_quant_8[:, 1::2] & 0xF) << 4)
    return x_quant_4, normalization.to(torch.float16)


def block_dequantize_4bit(x_quant_4: torch.Tensor, normalization: torch.Tensor) -> torch.Tensor:
    """
    The reverse operation of block_quantize_4bit.
    """
    assert x_quant_4.dim() == 2

    normalization = normalization.to(torch.float32)
    x_quant_8 = x_quant_4.new_empty(x_quant_4.size(0), x_quant_4.shape[1] * 2)
    x_quant_8[:, ::2] = x_quant_4 & 0xF
    x_quant_8[:, 1::2] = (x_quant_4 >> 4) & 0xF
    x_norm = x_quant_8.to(torch.float32) / 15
    x = (x_norm * 2 * normalization) - normalization
    return x.view(-1)


class Linear4Bit(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 16) -> None:
        super().__init__()
        self._shape = (out_features, in_features)
        self._group_size = group_size

        # Register buffers for quantized weights
        num_groups = (out_features * in_features + group_size - 1) // group_size
        self.register_buffer(
            "weight_q4",
            torch.zeros(num_groups, group_size // 2, dtype=torch.int8),
            persistent=False
        )
        self.register_buffer(
            "weight_norm",
            torch.zeros(num_groups, 1, dtype=torch.float16),
            persistent=False
        )

        # Register bias as buffer
        if bias:
            self.register_buffer('bias', torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_buffer('bias', None)

        self._register_load_state_dict_pre_hook(Linear4Bit._load_state_dict_pre_hook, with_module=True)

    def _load_state_dict_pre_hook(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if f"{prefix}weight" in state_dict:
            weight = state_dict[f"{prefix}weight"]
            del state_dict[f"{prefix}weight"]

            # Quantize weights
            weight = weight.view(-1)
            weight_q4, weight_norm = block_quantize_4bit(weight, group_size=self._group_size)
            
            # Store quantized weights
            self.weight_q4.copy_(weight_q4)
            self.weight_norm.copy_(weight_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize weights
        weight = block_dequantize_4bit(self.weight_q4, self.weight_norm)
        weight = weight.view(self._shape)
        
        # Create a view that requires gradients for forward pass
        weight = weight.detach().requires_grad_(True)
        
        return torch.nn.functional.linear(x, weight, self.bias)


class BigNet4Bit(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            self.model = torch.nn.Sequential(
                Linear4Bit(channels, channels),  # model.0
                torch.nn.GELU(),                # model.1
                Linear4Bit(channels, channels),  # model.2
                torch.nn.GELU(),                # model.3
                Linear4Bit(channels, channels),  # model.4
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self):
        super().__init__()
        # Six blocks with five LayerNorms between them
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM),      # model.0
            LayerNorm(BIGNET_DIM),       # model.1
            self.Block(BIGNET_DIM),      # model.2
            LayerNorm(BIGNET_DIM),       # model.3
            self.Block(BIGNET_DIM),      # model.4
            LayerNorm(BIGNET_DIM),       # model.5
            self.Block(BIGNET_DIM),      # model.6
            LayerNorm(BIGNET_DIM),       # model.7
            self.Block(BIGNET_DIM),      # model.8
            LayerNorm(BIGNET_DIM),       # model.9
            self.Block(BIGNET_DIM),      # model.10
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> BigNet4Bit:
    net = BigNet4Bit()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net
