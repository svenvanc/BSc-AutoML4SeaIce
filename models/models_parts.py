"""Models' components."""

# -- File info -- #
__author__ = ['Andreas R. Stokholm', 'Andrzej S. Kucik']
__copyright__ = ['Technical University of Denmark', 'European Space Agency']
__contact__ = ['stokholm@space.dtu.dk', 'andrzej.kucik@esa.int']
__version__ = '0.2.4'
__date__ = '2021-10-15'

# -- Third-party modules -- #
import torch
import torch.nn as nn


class FeatureMap(nn.Module):
    """Class to perform final 1D convolution before calculating cross entropy or using softmax."""

    def __init__(self, input_n, output_n):
        super(FeatureMap, self).__init__()

        self.feature_out = nn.Conv2d(input_n, output_n, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, x):
        """Pass x through final layer."""
        return self.feature_out(x)


class DoubleConv(nn.Module):
    """Class to perform a double conv layer in the U-NET architecture. Used in unet_model.py."""

    def __init__(self, options, input_n, output_n):
        super(DoubleConv, self).__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_n,
                      out_channels=output_n,
                      kernel_size=options['kernel_size'],
                      stride=options['stride_rate'],
                      padding=options['padding'],
                      padding_mode=options['padding_style'],
                      bias=False),
            nn.BatchNorm2d(output_n),
            nn.ReLU(),
            nn.Conv2d(in_channels=output_n,
                      out_channels=output_n,
                      kernel_size=options['kernel_size'],
                      stride=options['stride_rate'],
                      padding=options['padding'],
                      padding_mode=options['padding_style'],
                      bias=False),
            nn.BatchNorm2d(output_n),
            nn.ReLU()
        )

    def forward(self, x):
        """Pass x through the double conv layer."""
        x = self.double_conv(x)

        return x


class ContractingBlock(nn.Module):
    """Class to perform downward pass in the U-Net."""

    def __init__(self, options, input_n, output_n):
        super(ContractingBlock, self).__init__()

        self.contract_block = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.double_conv = DoubleConv(options, input_n, output_n)

    def forward(self, x):
        """Pass x through the downward layer."""
        x = self.contract_block(x)
        x = self.double_conv(x)
        return x


class ExpandingBlock(nn.Module):
    """Class to perform upward layer in the U-Net."""

    def __init__(self, options, input_n, output_n):
        super(ExpandingBlock, self).__init__()

        self.padding_style = options['padding_style']
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.double_conv = DoubleConv(options, input_n=input_n + output_n, output_n=output_n)

    def forward(self, x, x_skip):
        """Pass x through the upward layer and concatenate with opposite layer."""
        x = self.upsample(x)

        # Insure that x and skip H and W dimensions match.
        x = expand_padding(x, x_skip, padding_style=self.padding_style)
        x = torch.cat([x, x_skip], dim=1)

        return self.double_conv(x)




class MiddleDoubleConv(nn.Module):
    """Class to perform a double conv layer in the U-NET architecture. Used in models.py."""

    def __init__(self, options, feat_multiplier):
        super(MiddleDoubleConv, self).__init__()

        features = options['filters'][0] * feat_multiplier
        self.dropout = options['dropout']

        self.conv1 = nn.Conv2d(in_channels=features,
                               out_channels=features,
                               kernel_size=options['kernel_size'],
                               stride=options['stride_rate'],
                               padding=options['padding'],
                               padding_mode=options['padding_style'],
                               bias=False)
        self.bn1 = nn.BatchNorm2d(features)
        self.activation1 = nn.ReLU()
        self.dropout1 = nn.Dropout2d(options['dropout_rate']) if self.dropout else None

        self.conv2 = nn.Conv2d(in_channels=features,
                               out_channels=features,
                               kernel_size=options['kernel_size'],
                               stride=options['stride_rate'],
                               padding=options['padding'],
                               padding_mode=options['padding_style'],
                               bias=False)
        self.bn2 = nn.BatchNorm2d(features)
        self.activation2 = nn.ReLU()
        self.dropout2 = nn.Dropout2d(options['dropout_rate']) if self.dropout else None

    def forward(self, x):
        """Pass x through the double conv layer."""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        if self.dropout:
            x = self.dropout1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation2(x)
        if self.dropout:
            x = self.dropout2(x)

        return x


class ResidualBlock(nn.Module):
    """
    Residual block for the generator. Performs convolution, instance normalization (x2) and ads the
    output to the original input.

    Values
    ----------
    channels: int
            The number of input and output channels
    """

    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()

        # Layers
        self.conv1 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=(3, 3),
                               padding=(1, 1),
                               padding_mode='reflect',
                               bias=False)
        self.conv2 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=(3, 3),
                               padding=(1, 1),
                               padding_mode='reflect',
                               bias=False)
        self.instance_norm1 = nn.InstanceNorm2d(num_features=channels)
        self.instance_norm2 = nn.InstanceNorm2d(num_features=channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x :
            Image tensor of shape (batch size, channels, height, width).
        """
        input_x = x.clone()
        x = self.conv1(x)
        x = self.instance_norm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.instance_norm2(x)

        return input_x + x


def expand_padding(x, x_contract, padding_style: str = 'constant'):
    """
    Insure that x and x_skip H and W dimensions match.
    Parameters
    ----------
    x :
        Image tensor of shape (batch size, channels, height, width). Expanding path.
    x_contract :
        Image tensor of shape (batch size, channels, height, width) Contracting path.
        or torch.Size. Contracting path.
    padding_style : str
        Type of padding.

    Returns
    -------
    x : ndtensor
        Padded expanding path.
    """
    # Check whether x_contract is tensor or shape.
    if type(x_contract) == type(x):
        x_contract = x_contract.size()

    # Calculate necessary padding to retain patch size.
    pad_y = x_contract[2] - x.size()[2]
    pad_x = x_contract[3] - x.size()[3]

    if padding_style == 'zeros':
        padding_style = 'constant'

    x = torch.nn.functional.pad(x, [pad_x // 2, pad_x - pad_x // 2, pad_y // 2, pad_y - pad_y // 2], mode=padding_style)

    return x
