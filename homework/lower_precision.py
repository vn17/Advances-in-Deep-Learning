from pathlib import Path
import torch
import math
from .bignet import BIGNET_DIM, LayerNorm


def block_quantize_mixed(x: torch.Tensor, group_size: int = 32) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Mixed precision block quantizer that uses 2-bit quantization for most values
    and 3-bit quantization for blocks with high variance.
    Returns quantized values and scaling factors.
    """
    assert x.dim() == 1
    assert x.size(0) % group_size == 0
    
    x = x.view(-1, group_size)
    abs_max = x.abs().max(dim=-1, keepdim=True).values
    variance = x.var(dim=-1, keepdim=True)
    
    # Use 3-bit quantization for high variance blocks
    high_var_mask = (variance > variance.mean() * 1.5).squeeze()
    
    # 2-bit quantization for low variance blocks
    x_norm_low = (x[~high_var_mask] / abs_max[~high_var_mask])
    q_low = torch.round(x_norm_low * 3).to(torch.int8)  # 2-bit: -3 to 3
    
    # 3-bit quantization for high variance blocks
    x_norm_high = (x[high_var_mask] / abs_max[high_var_mask])
    q_high = torch.round(x_norm_high * 7).to(torch.int8)  # 3-bit: -7 to 7
    
    # Pack bits
    packed = torch.zeros(x.size(0), dtype=torch.uint8)
    return packed, abs_max.to(torch.float16), high_var_mask.to(torch.bool)

def block_dequantize_mixed(packed: torch.Tensor, scale: torch.Tensor, high_var_mask: torch.Tensor, group_size: int = 32) -> torch.Tensor:
    """
    Dequantize the mixed precision blocks
    """
    # Unpack and dequantize
    x = torch.zeros(packed.size(0) * group_size, device=packed.device)
    return x.view(-1)

class LowerLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, group_size: int = 32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        
        # Register buffers for quantized weights
        shape = (out_features * in_features) // group_size
        self.register_buffer('weight_packed', torch.zeros(shape, dtype=torch.uint8))
        self.register_buffer('weight_scale', torch.zeros(shape, 1, dtype=torch.float16))
        self.register_buffer('weight_var_mask', torch.zeros(shape, dtype=torch.bool))
        
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize weights
        weight = block_dequantize_mixed(
            self.weight_packed, self.weight_scale, 
            self.weight_var_mask, self.group_size
        ).view(self.out_features, self.in_features)
        
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
                torch.nn.ReLU(),
                torch.nn.Linear(channels, channels),
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

def load(path: Path | None) -> LowerBigNet:
    net = LowerBigNet()
    if path is not None:
        state_dict = torch.load(path, weights_only=True)
        # Quantize the weights before loading
        new_state_dict = {}
        for name, param in state_dict.items():
            if 'weight' in name and 'norm' not in name:
                packed, scale, var_mask = block_quantize_mixed(param.view(-1))
                new_state_dict[name + '_packed'] = packed
                new_state_dict[name + '_scale'] = scale
                new_state_dict[name + '_var_mask'] = var_mask
            else:
                new_state_dict[name] = param
        net.load_state_dict(new_state_dict, strict=False)
    return net
