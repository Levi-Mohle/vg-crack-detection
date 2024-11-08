from src.models.components.blocks.normflow_blocks import *


class NormFlowMultiScaleNet(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        num_blocks: int = 4,
        c_hidden_1: int = 16,
        c_hidden_2: int = 32,
        c_hidden_3: int = 48,
        c_hidden_4: int = 64
    ) -> None:
        
        super().__init__()

        flow_layers = []

        vardeq_layers = [
            CouplingLayer(
                network=GatedConvNet(c_in=2, c_out=2, c_hidden=16),
                mask=create_checkerboard_mask(h=28, w=28, invert=(i % 2 == 1)),
                c_in=1,
            )
            for i in range(4)
        ]
        flow_layers += [VariationalDequantization(vardeq_layers)]

        flow_layers += [
            CouplingLayer(
                network=GatedConvNet(c_in=1, c_hidden=32),
                mask=create_checkerboard_mask(h=28, w=28, invert=(i % 2 == 1)),
                c_in=1,
            )
            for i in range(2)
        ]
        flow_layers += [SqueezeFlow()]
        for i in range(2):
            flow_layers += [
                CouplingLayer(
                    network=GatedConvNet(c_in=4, c_hidden=48), mask=create_channel_mask(c_in=4, invert=(i % 2 == 1)), c_in=4
                )
            ]
        flow_layers += [SplitFlow(), SqueezeFlow()]
        for i in range(4):
            flow_layers += [
                CouplingLayer(
                    network=GatedConvNet(c_in=8, c_hidden=64), mask=create_channel_mask(c_in=8, invert=(i % 2 == 1)), c_in=8
                )
            ]

        self.flows = flow_layers
        self.model = nn.ModuleList(flow_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """

        return self.model(x)
        
if __name__ == "__main__":
    _ = NormFlowMultiScaleNet()