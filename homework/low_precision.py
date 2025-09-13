from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401


def block_quantize_4bit(x: torch.Tensor, group_size: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize the input tensor to 4-bit precision along the last dimension.
    Always quantize group_size value together and store their absolute value first.
    To keep things simple, we require x to be a 1D tensor, and the size divisible by group_size.
    Return the quantized tensor and scaling factor.
    """
    assert x.dim() == 1
    assert x.size(0) % group_size == 0

    x = x.view(-1, group_size)
    
    # Use max absolute value for better range utilization
    scale = x.abs().max(dim=-1, keepdim=True).values
    scale = torch.clamp(scale, min=1e-8)  # Prevent division by zero
    
    # Normalize to [-1, 1] range
    x_norm = x / scale
    
    # Quantize to 4-bit: map [-1, 1] to [0, 15]
    x_quant_8 = torch.clamp((x_norm * 7.5 + 7.5).round(), 0, 15).to(torch.int8)
    
    # Pack two 4-bit values into one 8-bit value
    x_quant_4 = (x_quant_8[:, ::2] & 0xF) + ((x_quant_8[:, 1::2] & 0xF) << 4)
    return x_quant_4, scale.to(torch.float16)


def block_dequantize_4bit(x_quant_4: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """
    The reverse operation of block_quantize_4bit.
    """
    assert x_quant_4.dim() == 2

    scale = scale.to(torch.float32)
    
    # Unpack 4-bit values
    x_quant_8 = x_quant_4.new_empty(x_quant_4.size(0), x_quant_4.shape[1] * 2, dtype=torch.int8)
    x_quant_8[:, ::2] = x_quant_4 & 0xF
    x_quant_8[:, 1::2] = (x_quant_4 >> 4) & 0xF
    
    # Dequantize: map [0, 15] back to [0, 1]
    x_norm = x_quant_8.to(torch.float32) / 15.0
    
    # Map back to original range: reverse of (x + scale) / (2 * scale)
    x = x_norm * (2 * scale) - scale
    return x.view(-1)


class Linear4Bit(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 256) -> None:
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
            "weight_scale",
            torch.ones(num_groups, 1, dtype=torch.float16),
            persistent=False
        )

        # Register bias for the layer (no learnable quantization bias)
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
            weight_q4, weight_scale = block_quantize_4bit(weight, group_size=self._group_size)
            
            # Store quantized weights
            self.weight_q4.copy_(weight_q4)
            self.weight_scale.copy_(weight_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize weights
        weight_deq = block_dequantize_4bit(self.weight_q4, self.weight_scale)
        weight_deq = weight_deq.view(self._shape)

        return torch.nn.functional.linear(x, weight_deq, self.bias)


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
        state = torch.load(path, weights_only=True)
        net.load_state_dict(state, strict=False)
    return net