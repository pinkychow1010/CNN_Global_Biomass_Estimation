import torch
from torch import nn
import numpy as np


def depthwise_separable_convolution_block(
    in_channels: int, intermediate_depth: int, kernel_size: int, padding: str = "same"
):
    return nn.Sequential(
        nn.ReLU(),
        # Convolve every feature map with its own set of features
        nn.Conv2d(
            intermediate_depth,
            intermediate_depth,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=1,
            padding=padding,
        ),
        # Convolve the feature maps along the depth dimenion using a 1x1 kernel
        nn.Conv2d(
            intermediate_depth,
            intermediate_depth,
            kernel_size=1,
            stride=1,
            padding=padding,
        ),
        nn.BatchNorm2d(intermediate_depth),
        nn.ReLU(),
        nn.Conv2d(
            intermediate_depth,
            intermediate_depth,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=1,
            padding=padding,
        ),
        nn.Conv2d(
            intermediate_depth,
            intermediate_depth,
            kernel_size=1,
            stride=1,
            padding=padding,
        ),
        nn.BatchNorm2d(intermediate_depth),
    )


class BaseNeuralNetwork(nn.Module):
    """Base model based on Nico Langs work.

    Base implementation of the network described by Lang et al., 2019 'Country-wide high-resolution vegetation height mapping with Sentinel-2' (https://arxiv.org/pdf/1904.13270.pdf)
    with the modifications described in Lang et al., 2022 'A high-resolution canopy height model of the Earth' (https://arxiv.org/pdf/2204.08322.pdf)
    to reduce the model size and increase training speed.

    Summary of the changes between the two papers:
        - Number of blocks: 8 (from 18)
        - Number of filters per block: 256 (from 728)

    Args:
        in_channels: Number of input channels of the input data
        intermediate_depth: Number of filters used by the depthwise separable blocks (paper default: 256)
        kernel_size: Size of the kernel used by the epthwise separable blocks (paper default: 3x3)
        padding: Padding to use (default: 'same')
    """

    def __init__(
        self,
        in_channels: int,
        intermediate_depth: int = 224,
        kernel_size: int = 3,
        padding: str = "same",
        output_variance=False,
    ):
        super(BaseNeuralNetwork, self).__init__()

        if intermediate_depth % in_channels != 0:
            raise ValueError(
                f"'intermediate_depth' ({intermediate_depth}) must be divisible by 'in_channels' ({in_channels}) for separable convolution to work!"
            )

        # Entry block: Gradually increase the depth to 256 (intermediate_depth)
        self.entry_block_conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1, stride=1, padding=padding),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(
                128, intermediate_depth, kernel_size=1, stride=1, padding=padding
            ),
            nn.BatchNorm2d(intermediate_depth),
            nn.ReLU(),
        )
        self.input_residuals = nn.Conv2d(
            in_channels, intermediate_depth, kernel_size=1, stride=1, padding=padding
        )
        # Depthwise separable convolution
        self.depth_sep_conv1 = depthwise_separable_convolution_block(
            intermediate_depth, intermediate_depth, kernel_size, padding
        )
        self.depth_sep_conv2 = depthwise_separable_convolution_block(
            intermediate_depth, intermediate_depth, kernel_size, padding
        )
        self.depth_sep_conv3 = depthwise_separable_convolution_block(
            intermediate_depth, intermediate_depth, kernel_size, padding
        )
        self.depth_sep_conv4 = depthwise_separable_convolution_block(
            intermediate_depth, intermediate_depth, kernel_size, padding
        )
        self.depth_sep_conv5 = depthwise_separable_convolution_block(
            intermediate_depth, intermediate_depth, kernel_size, padding
        )
        self.depth_sep_conv6 = depthwise_separable_convolution_block(
            intermediate_depth, intermediate_depth, kernel_size, padding
        )
        self.depth_sep_conv7 = depthwise_separable_convolution_block(
            intermediate_depth, intermediate_depth, kernel_size, padding
        )
        self.depth_sep_conv8 = depthwise_separable_convolution_block(
            intermediate_depth, intermediate_depth, kernel_size, padding
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(intermediate_depth, 1, kernel_size=1, stride=1, padding=padding),
            # attempt to resolve dying relu problem
            # using both leakyReLU & ELU
            # 1) significantly speed up converge time
            # 2) increase model training stability
            # 3) reduce overfitting
            nn.LeakyReLU(),
            nn.ELU(),
            # nn.ReLU(),  # test: allow only positive biomass prediction
            # (issue: https://discuss.pytorch.org/t/why-my-predictions-are-all-zeros-during-the-training-phase-of-my-neural-network/119991/2)
        )
        self.variance = nn.Sequential(
            nn.Conv2d(
                intermediate_depth, 1, kernel_size=1, stride=1, padding=padding
            )  # ReLU?
        )

        self.output_variance = output_variance

    def forward(self, x):
        # Entry block
        x1 = self.entry_block_conv(x)
        r1 = self.input_residuals(x)
        x2 = torch.add(r1, x1)

        # Depthwise separable convolution blocks
        x3 = torch.add(self.depth_sep_conv1(x2), x2)
        x4 = torch.add(self.depth_sep_conv2(x3), x3)
        x5 = torch.add(self.depth_sep_conv3(x4), x4)
        x6 = torch.add(self.depth_sep_conv4(x5), x5)
        x7 = torch.add(self.depth_sep_conv5(x6), x6)
        x8 = torch.add(self.depth_sep_conv6(x7), x7)
        x9 = torch.add(self.depth_sep_conv7(x8), x8)
        x10 = torch.add(self.depth_sep_conv8(x9), x9)

        out = self.output_conv(x10)

        if self.output_variance:
            # Make sure that the variance is positive
            var = torch.abs(self.variance(x10))
            out = torch.concat([out, var], axis=1)
        return out


if __name__ == "__main__":
    model = BaseNeuralNetwork(in_channels=14)
    test_input = torch.Tensor(np.random.rand(1, 14, 16, 16))
    print(model.forward)
    print(model(test_input).shape)
    print(
        f"Number of parameters: {sum([param.nelement() for param in model.parameters()])}"
    )
