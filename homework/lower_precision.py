from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm


def block_quantize_3bit(x: torch.Tensor, group_size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 1
    assert x.size(0) % group_size == 0

    x = x.view(-1, group_size)
    normalization = x.abs().max(dim=-1, keepdim=True).values
    x_norm = (x + normalization) / (2 * normalization)
    x_quant_8 = (x_norm * 7).round().to(torch.int8)  # 3 bits → 8 levels
    # Pack 3-bit values into bytes (up to 2 per byte, leaving 2 bits unused per byte)
    x_quant_3 = (x_quant_8[:, ::2] & 0x7) + ((x_quant_8[:, 1::2] & 0x7) << 3)
    return x_quant_3, normalization.to(torch.float16)


def block_dequantize_3bit(x_quant_3: torch.Tensor, normalization: torch.Tensor) -> torch.Tensor:
    assert x_quant_3.dim() == 2

    normalization = normalization.to(torch.float32)
    x_quant_8 = x_quant_3.new_empty(x_quant_3.size(0), x_quant_3.shape[1] * 2)
    x_quant_8[:, ::2] = x_quant_3 & 0x7
    x_quant_8[:, 1::2] = (x_quant_3 >> 3) & 0x7
    x_norm = x_quant_8.to(torch.float32) / 7
    x = (x_norm * 2 * normalization) - normalization
    return x.view(-1)


class Linear3Bit(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 16) -> None:
        super().__init__()
        self._shape = (out_features, in_features)
        self._group_size = group_size

        self.register_buffer(
            "weight_q3",
            torch.zeros(out_features * in_features // group_size, group_size // 2, dtype=torch.int8),
            persistent=False,
        )
        self.register_buffer(
            "weight_norm",
            torch.zeros(out_features * in_features // group_size, 1, dtype=torch.float16),
            persistent=False,
        )

        self._register_load_state_dict_pre_hook(Linear3Bit._load_state_dict_pre_hook, with_module=True)
        self.bias = torch.nn.Parameter(torch.zeros(out_features, dtype=torch.float16)) if bias else None

    def _load_state_dict_pre_hook(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        key = f"{prefix}weight"
        if key in state_dict:
            weight = state_dict[key]
            del state_dict[key]
            weight_flat = weight.flatten()
            pad_len = (self._group_size - (weight_flat.numel() % self._group_size)) % self._group_size
            if pad_len > 0:
                weight_flat = torch.cat([weight_flat, torch.zeros(pad_len, device=weight_flat.device, dtype=weight_flat.dtype)])
            q, norm = block_quantize_3bit(weight_flat, self._group_size)
            self.weight_q3.copy_(q)
            self.weight_norm.copy_(norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            weight = block_dequantize_3bit(self.weight_q3, self.weight_norm).to(torch.float16)
            weight = weight[:self._shape[0]*self._shape[1]]  # truncate padded elements
            weight = weight.view(self._shape)
            x = x.to(torch.float16)
        return torch.nn.functional.linear(x, weight, self.bias)


class BigNet3Bit(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Sequential(
                Linear3Bit(1024, 1024),
                torch.nn.ReLU(),
                Linear3Bit(1024, 1024),
                torch.nn.ReLU(),
                Linear3Bit(1024, 1024),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.model(x) + x

    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(),
            LayerNorm(1024, dtype=torch.float16),
            self.Block(),
            LayerNorm(1024, dtype=torch.float16),
            self.Block(),
            LayerNorm(1024, dtype=torch.float16),
            self.Block(),
            LayerNorm(1024, dtype=torch.float16),
            self.Block(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float16)
        return self.model(x)


def load(path: Path | None) -> BigNet3Bit:
    net = BigNet3Bit()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net