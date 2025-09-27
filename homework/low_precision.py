from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm


def block_quantize_4bit(x: torch.Tensor, group_size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 1
    assert x.size(0) % group_size == 0

    x = x.view(-1, group_size)
    normalization = x.abs().max(dim=-1, keepdim=True).values
    x_norm = (x + normalization) / (2 * normalization)
    x_quant_8 = (x_norm * 15).round().to(torch.int8)
    x_quant_4 = (x_quant_8[:, ::2] & 0xF) + ((x_quant_8[:, 1::2] & 0xF) << 4)
    return x_quant_4, normalization.to(dtype=torch.float16, device=x.device)


def block_dequantize_4bit(x_quant_4: torch.Tensor, normalization: torch.Tensor) -> torch.Tensor:
    assert x_quant_4.dim() == 2

    normalization = normalization.to(dtype=torch.float32, device=x_quant_4.device)
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

        self.register_buffer(
            "weight_q4",
            torch.zeros(out_features * in_features // group_size, group_size // 2, dtype=torch.int8),
            persistent=False,
        )
        self.register_buffer(
            "weight_norm",
            torch.zeros(out_features * in_features // group_size, 1, dtype=torch.float16),
            persistent=False,
        )

        self._register_load_state_dict_pre_hook(Linear4Bit._load_state_dict_pre_hook, with_module=True)
        self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float32)) if bias else None

    def _load_state_dict_pre_hook(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        key = f"{prefix}weight"
        if key in state_dict:
            weight = state_dict[key]
            del state_dict[key]
            # Flatten, quantize, and store
            weight_flat = weight.flatten()
            q, norm = block_quantize_4bit(weight_flat, self._group_size)
            self.weight_q4.copy_(q)
            self.weight_norm.copy_(norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            weight = block_dequantize_4bit(self.weight_q4, self.weight_norm)
            weight = weight.view(self._shape)
        return torch.nn.functional.linear(x, weight, self.bias)


class BigNet4Bit(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.model = torch.nn.Sequential(
                Linear4Bit(channels, channels),
                torch.nn.ReLU(),
                Linear4Bit(channels, channels),
                torch.nn.ReLU(),
                Linear4Bit(channels, channels),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> BigNet4Bit:
    net = BigNet4Bit()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net