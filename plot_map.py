"""
This file is to be used as a template for creating custom map plots. The file contains functions for plotting both
.nc files and .npy files. At the bottum of the script some example functions are given of how to mask out certain values
for a custom plot.
"""

# -- Third-party modules -- #
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import colors

# --Proprietary modules -- #
from init_laptop import OPTIONS

# Names of the SIC classes.
SIC_GROUPS = {
    0: 0,
    1: 10,
    2: 20,
    3: 30,
    4: 40,
    5: 50,
    6: 60,
    7: 70,
    8: 80,
    9: 90,
    10: 100,
    11: 'unknown'  # Added
}

ICE_STRINGS = {
    'SIC': 'Sea Ice Concentration [%]',
    'SOD': 'Stage of Development',
    'FLOE': 'Floe Size'
}


def plotmap_ESA2(values, modelname):
    '''A plotting function based on .nc file inputs'''

    fig, axs = plt.subplots(nrows=1, ncols=2,
                            # figsize=(12, 8)
                            )

    axs[0].imshow(values, vmin=OPTIONS['vmin'][OPTIONS['chart']], vmax=OPTIONS['vmax'][OPTIONS['chart']],
               cmap=OPTIONS['cmap'][OPTIONS['chart']])

    axs[0].set_xticks([])
    axs[0].set_yticks([])

    axs[1].imshow(values, vmin=OPTIONS['vmin'][OPTIONS['chart']], vmax=OPTIONS['vmax'][OPTIONS['chart']],
               cmap=OPTIONS['cmap'][OPTIONS['chart']])

    axs[1].set_xticks([])
    axs[1].set_yticks([])

    plt.subplots_adjust(wspace=0.05)

    arranged = np.arange(0, len(SIC_GROUPS.values()))
    ncolor = OPTIONS['cmap'][OPTIONS['chart']].N
    norm = colors.BoundaryNorm(arranged - 0.5, ncolor)  # Get colour boundaries. -0.5 to center ticks for each color.

    cmap = OPTIONS['cmap'][OPTIONS['chart']]
    cmap = cmap.with_extremes(over='white')  # class 11 ==> unknown
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                      ticks=arranged,
                      location='bottom',
                      ax=axs,
                      pad=0.05,
                      # shrink=0.8,
                      fraction=0.06
                      )
    cb.set_label(label=ICE_STRINGS['SIC'])
    cb.set_ticklabels(list(SIC_GROUPS.values()))

    # dpi = 512
    dpi = 128

    fig.savefig(modelname,
                       dpi=dpi,
                       format='png',
                       bbox_inches='tight',
                       pad_inches=0.05,
                       transparent=False)
    plt.close('all')


def apply_unknown_mask_to_y(x, y):
    '''This function masks y values to class 11 (land values) that have corresponding x values which are invalid'''

    mask = x == 0.0
    mask = mask[:, :, :, 0:1]
    # mask = mask[:, :, :, 1:2]
    mask = np.squeeze(mask)
    y[mask] = 11


def plot_scene(x, y, prediction, model_name):
    '''A plotting function based on .npy file inputs. Plots a scene given input x, and target y, a prediction y and a name.'''

    fig, axs = plt.subplots(nrows=1, ncols=2)

    if x is not None:
        apply_unknown_mask_to_y(x, y)
        apply_unknown_mask_to_y(x, prediction)

    cmap = OPTIONS['cmap'][OPTIONS['chart']]
    cmap.set_under("black")
    cmap.set_over("white")

    axs[0].imshow(y, vmin=OPTIONS['vmin'][OPTIONS['chart']], vmax=OPTIONS['vmax'][OPTIONS['chart']],
                  cmap=cmap)
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    axs[1].imshow(prediction, vmin=OPTIONS['vmin'][OPTIONS['chart']], vmax=OPTIONS['vmax'][OPTIONS['chart']],
                  cmap=cmap)
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    plt.subplots_adjust(wspace=0.05)
    arranged = np.arange(0, len(SIC_GROUPS.values()))
    ncolor = OPTIONS['cmap'][OPTIONS['chart']].N
    norm = colors.BoundaryNorm(arranged - 0.5, ncolor)  # Get colour boundaries. -0.5 to center ticks for each color.

    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=OPTIONS['cmap'][OPTIONS['chart']]),
                      ticks=arranged,
                      location='bottom',
                      ax=axs,
                      pad=0.05,
                      # shrink=0.8,
                      fraction=0.06
                      )
    cb.set_label(label=ICE_STRINGS['SIC'])
    cb.set_ticklabels(list(SIC_GROUPS.values()))

    fig.savefig(model_name,
                dpi=128,
                format='png',
                bbox_inches='tight',
                pad_inches=0.05,
                transparent=False)
    plt.close('all')


def plot_nc_example():
    '''An example function that plots a map based on a .nc file'''

    scene_name = '20181217T210415_pro.nc'
    model_name = "./example.png"

    ds_train = xr.open_dataset(OPTIONS['path_to_processed_data'] + scene_name)
    values = ds_train['SIC'].values
    cond = values == 11  # in general, a boolean expression is required
    values[cond] = float('NaN')
    plotmap_ESA2(values, model_name)


def plot_npy_example():
    '''An example function that plots a map based on a .npy file'''

    model_name = "./example.png"
    name_x = '20181212T205512-nersc-x.npy'
    name_y = '20181212T205512-y.npy'

    y = np.load(name_y)
    y2 = np.load(name_y)
    x = np.load(name_x)

    # cond is a mask where all values are 0.0 for the first sar value
    cond = x == 0.0
    cond = cond[:, :, 0]

    # cond2 is a mask where all values are 0.0 for the second sar value
    cond2 = x == 0.0
    cond2 = cond2[:, :, 1]

    # All values where either of the conditions are true will be set to a value outside the range of 0-100
    y[cond] = 110.0
    y2[cond2] = 110.0

    plot_scene(None, y, y2, model_name)


if __name__ == '__main__':
    plot_nc_example()
    plot_npy_example()


