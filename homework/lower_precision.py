from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm


def block_quantize_4bit(x: torch.Tensor, group_size: int = 1048576) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.to(torch.float16)
    assert x.dim() == 1
    assert x.size(0) % group_size == 0

    x = x.view(-1, group_size)
    normalization = x.abs().max(dim=-1, keepdim=True).values
    x_norm = x
    x_norm.add_(normalization)           # in-place addition
    x_norm.div_(2 * normalization)       # in-place division
    x_quant_16 = (x_norm * 15).round().to(torch.int8)  # 4 bits → 16 levels
    # Pack 4-bit values into bytes (2 per byte)
    x_quant_4 = (x_quant_16[:, ::2] & 0xF) + ((x_quant_16[:, 1::2] & 0xF) << 4)
    return x_quant_4, normalization





class Linear4Bit(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 1048576) -> None:
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
        # Preallocate a temporary buffer for dequantization
        self.register_buffer("_tmp_dequant", torch.empty(0, dtype=torch.int8), persistent=False)

        self._register_load_state_dict_pre_hook(Linear4Bit._load_state_dict_pre_hook, with_module=True)
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
            q, norm = block_quantize_4bit(weight_flat, self._group_size)
            self.weight_q4.copy_(q)
            self.weight_norm.copy_(norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            # reuse tmp buffer for dequantization, no extra allocations
            needed_numel = self.weight_q4.size(0) * self.weight_q4.size(1) * 2
            if self._tmp_dequant.numel() < needed_numel:
                self._tmp_dequant.resize_(self.weight_q4.size(0), self.weight_q4.size(1) * 2)
            x_quant_16 = self._tmp_dequant[:self.weight_q4.size(0), :self.weight_q4.size(1)*2]
            x_quant_16[:, ::2] = self.weight_q4 & 0xF
            x_quant_16[:, 1::2] = (self.weight_q4 >> 4) & 0xF
            x_norm = x_quant_16.to(torch.float16) / 15
            weight = (x_norm * 2 * self.weight_norm.view(-1,1)) - self.weight_norm.view(-1,1)
            weight = weight[:self._shape[0]*self._shape[1]].view(self._shape)
            # assume x is already float16 from BigNet4Bit.forward
        return torch.nn.functional.linear(x, weight, self.bias)


class BigNet4Bit(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Sequential(
                Linear4Bit(1024, 1024),
                torch.nn.ReLU(inplace=True),
                Linear4Bit(1024, 1024),
                torch.nn.ReLU(inplace=True),
                Linear4Bit(1024, 1024),
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
            LayerNorm(1024, dtype=torch.float16),
            self.Block(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float16, copy=False)
        return self.model(x)


def load(path: Path | None) -> BigNet4Bit:
    net = BigNet4Bit()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True))
    return net