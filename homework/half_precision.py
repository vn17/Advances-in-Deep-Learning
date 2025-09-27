from pathlib import Path

import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401


class HalfLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
    ) -> None:
        """
        Implement a half-precision Linear Layer.
        Feel free to use the torch.nn.Linear class as a parent class (it makes load_state_dict easier, names match).
        Feel free to set self.requires_grad_ to False, we will not backpropagate through this layer.
        """
        super().__init__(in_features, out_features, bias)
        # Cast weights and biases to half precision (float16) and move to DEVICE
        self.weight.data = self.weight.data.half().to(DEVICE)
        if self.bias is not None:
            self.bias.data = self.bias.data.half().to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Move input to correct device if needed
        if x.device != self.weight.device:
            x = x.to(self.weight.device)
        # Perform computation in float16
        x_fp16 = x.half() if x.dtype != torch.float16 else x
        output = torch.nn.functional.linear(x_fp16, self.weight, self.bias)
        return output  # Keep output in float16 to reduce memory


class HalfBigNet(torch.nn.Module):
    """
    A BigNet where all weights are in half precision. Make sure that the normalization uses full
    precision though to avoid numerical instability.
    """

    class Block(torch.nn.Module):
        def __init__(self, channels: int):
            super().__init__()
            # TODO: Implement me (feel free to copy and reuse code from bignet.py)
            self.model = torch.nn.Sequential(
                HalfLinear(channels, channels).to(DEVICE),
                torch.nn.ReLU(),
                HalfLinear(channels, channels).to(DEVICE),
                torch.nn.ReLU(),
                HalfLinear(channels, channels).to(DEVICE),
            ).to(DEVICE)

        def forward(self, x: torch.Tensor):
            out = self.model(x)
            x.add_(out)  # in-place add to reduce memory
            return x

    def __init__(self):
        super().__init__()
        # TODO: Implement me (feel free to copy and reuse code from bignet.py)
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM).to(DEVICE),
            LayerNorm(BIGNET_DIM).to(DEVICE),
            self.Block(BIGNET_DIM).to(DEVICE),
            LayerNorm(BIGNET_DIM).to(DEVICE),
            self.Block(BIGNET_DIM).to(DEVICE),
            LayerNorm(BIGNET_DIM).to(DEVICE),
            self.Block(BIGNET_DIM).to(DEVICE),
            LayerNorm(BIGNET_DIM).to(DEVICE),
            self.Block(BIGNET_DIM).to(DEVICE),
            LayerNorm(BIGNET_DIM).to(DEVICE),
            self.Block(BIGNET_DIM).to(DEVICE),
        ).to(DEVICE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Move input to correct device once at start and reuse
        if x.device != DEVICE:
            x = x.to(DEVICE)
        return self.model(x)


def load(path: Path | None) -> HalfBigNet:
    # You should not need to change anything here
    # PyTorch can load float32 states into float16 models
    net = HalfBigNet().to(DEVICE)
    if path is not None:
        net.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    return net
