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
        # Hint: Use the .to method to cast a tensor to a different dtype (i.e. torch.float16 or x.dtype)
        # The input and output should be of x.dtype = torch.float32
        x = x.to(torch.float32).to(next(self.parameters()).device)  # Ensure input is in float32 and on correct device
        output = torch.nn.functional.linear(
            x.to(torch.float16), self.weight, self.bias
        )  # Perform computation in float16
        return output.to(torch.float32).to(x.device)  # Cast output back to float32 and move to input device


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
            return self.model(x) + x

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
        return self.model(x)


def load(path: Path | None) -> HalfBigNet:
    # You should not need to change anything here
    # PyTorch can load float32 states into float16 models
    net = HalfBigNet().to(DEVICE)
    if path is not None:
        net.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    return net
