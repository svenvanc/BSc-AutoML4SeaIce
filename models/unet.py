"""UNet model."""

# -- File info -- #
__author__ = 'Andreas R. Stokholm'
__contributor__ = 'Andrzej S. Kucik'
__copyright__ = ['Technical University of Denmark', 'European Space Agency']
__contact__ = ['stokholm@space.dtu.dk', 'andrzej.kucik@esa.int']
__version__ = '0.2.5'
__date__ = '2021-11-04'

# -- Third-party modules -- #
from torch import nn as nn

# -- Proprietary modules -- #
from models.models_parts import DoubleConv, ContractingBlock, ExpandingBlock, FeatureMap


class UNet(nn.Module):
    """PyTorch U-Net Class. Uses unet_parts."""

    def __init__(self, options):
        super().__init__()

        self.input_block = DoubleConv(options, input_n=options['train_variables'].size, output_n=options['filters'][0])

        self.contract_blocks = nn.ModuleList()
        for contract_n in range(1, len(options['filters'])):
            self.contract_blocks.append(
                ContractingBlock(options=options,
                                 input_n=options['filters'][contract_n - 1],
                                 output_n=options['filters'][contract_n]))  # only used to contract input patch.

        self.bridge = ContractingBlock(options, input_n=options['filters'][-1], output_n=options['filters'][-1])

        self.expand_blocks = nn.ModuleList()
        self.expand_blocks.append(
            ExpandingBlock(options=options, input_n=options['filters'][-1], output_n=options['filters'][-1]))

        for expand_n in range(len(options['filters']), 1, -1):
            self.expand_blocks.append(ExpandingBlock(options=options,
                                                     input_n=options['filters'][expand_n - 1],
                                                     output_n=options['filters'][expand_n - 2]))

        # Number of output parameters of the model depending on loss function. Classification vs Regression.
        if options['loss'].lower() == 'emd2' or options['loss'].lower() == 'ce':
            output_n = options['n_classes'][options['chart']]
        else:
            output_n = 1

        self.feature_map = FeatureMap(input_n=options['filters'][0], output_n=output_n)

    def forward(self, x):
        """Forward model pass."""
        x_contract = [self.input_block(x)]
        for contract_block in self.contract_blocks:
            x_contract.append(contract_block(x_contract[-1]))
        x_expand = self.bridge(x_contract[-1])
        up_idx = len(x_contract)
        for expand_block in self.expand_blocks:
            x_expand = expand_block(x_expand, x_contract[up_idx - 1])
            up_idx -= 1

        return self.feature_map(x_expand)
