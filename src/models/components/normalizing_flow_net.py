from src.models.components.blocks.normflow_blocks import *

class NormFlowNet(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        use_vardeq: bool = True,
        num_blocks: int = 8,
        c_hidden: int = 32
    ) -> None:
        """Initialize a `NormflowNet` module.

        """
        super().__init__()

        flow_layers = []
        if use_vardeq:
            vardeq_layers = [
                CouplingLayer(
                    network=GatedConvNet(c_in=2, c_out=2, c_hidden=c_hidden//2),
                    mask=create_checkerboard_mask(h=28, w=28, invert=(i % 2 == 1)),
                    c_in=1,
                )
                for i in range(num_blocks//2)
            ]
            flow_layers += [VariationalDequantization(var_flows=vardeq_layers)]
        else:
            flow_layers += [Dequantization()]

        for i in range(num_blocks):
            flow_layers += [
                CouplingLayer(
                    network=GatedConvNet(c_in=1, c_hidden=c_hidden),
                    mask=create_checkerboard_mask(h=28, w=28, invert=(i % 2 == 1)),
                    c_in=1,
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
    _ = NormFlowNet()