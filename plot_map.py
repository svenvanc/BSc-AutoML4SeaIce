"""
Calculated the mean of the training files, then uses this mean (and the rounded value of this mean)
as a prediction for the validate files.

Files are read using the data paths set in init.py

Prints the following statistics:
    - R2
    - RMSE
    - rounded R2 (uses rounded mean)
    - rounded RMSE (uses rounded mean)
    - accuracy (uses rounded mean)
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

def plotmap_ESA2(values):
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

    # cbar = plt.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=OPTIONS['cmap'][OPTIONS['chart']]), ticks=arranged, fraction=0.0485, pad=0.049, ax=ax)
    # cbar.set_label(label=OPTIONS['ICE_STRINGS']['SIC'])
    # cbar.set_ticklabels(list(SIC_GROUPS.values()))

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

    # plt.title(ICE_STRINGS['SIC'])

    # cb = plt.cm.ScalarMappable(norm=norm, cmap=OPTIONS['cmap'][OPTIONS['chart']])
    # cb.set_clim((0., 100.))
    # plt.gcf().colorbar(cb, location="bottom", drawedges="True")

    # title = 'Sea Ice Concentration [%]'
    # plt.title(title)

    # show_pixel_count(options)
    # show_chart_colorbar(options, options['chart'])

    # dpi = 512
    dpi = 128

    modelname = "./test.png"
    fig.savefig(modelname,
                       dpi=dpi,
                       format='png',
                       bbox_inches='tight',
                       pad_inches=0.05,
                       transparent=False)
    plt.close('all')


def apply_unknown_mask_to_y(x, y):
    mask = x == 0.0
    mask = mask[:, :, :, 0:1]
    mask = np.squeeze(mask)

    y[mask] = 11


def apply_unknown_mask_to_y_to_255(x, y):
    mask = x == 0.0
    mask = mask[:, :, :, 0:1]
    mask = np.squeeze(mask)

    y[mask] = 255


def save_prediction(prediction, land_values_mask, name, column_size):
    prediction = prediction.astype(float)
    prediction[land_values_mask] = float('NaN')

    row = int(len(prediction) / column_size)
    col = column_size
    prediction_2D = np.reshape(prediction, (row, col))

    file = open("/home/s2358093/data1/tmp/" + name + ".txt", "w+")

    np.savetxt(file, prediction_2D)
    file.close()
    return


def plot_scene(x, y, prediction, name):
    fig, axs = plt.subplots(nrows=1, ncols=2)

    # apply_unknown_mask_to_y(x, y)
    # apply_unknown_mask_to_y(x, prediction)

    cmap = OPTIONS['cmap'][OPTIONS['chart']]
    # cmap = cmap.set_extremes(under="black", over="white")
    # cmap = cmap.with_extremes(over='white')  # class 11 ==> unknown
    # cmap = cmap.with_extremes(under='black')  # class 11 ==> unknown
    cmap.set_under("black")
    cmap.set_over("white")
    # cmap.set_bad('black')
    # cmap.set_over(110.0)
    # cmap.set_under(120.0)

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

    # model_name = os.path.join(path_to_image_storing, name + '.png')
    model_name = "./test_1.png"
    fig.savefig(model_name,
                dpi=128,
                format='png',
                bbox_inches='tight',
                pad_inches=0.05,
                transparent=False)
    plt.close('all')



if __name__ == '__main__':
    # # name = '20180910T081714_pro.nc'
    # name = '20181217T210415_pro.nc'
    # name = '20181212T205512-y.npy'
    # ds_train = xr.open_dataset('/home/svenvanc/Universiteit_Leiden/Bachelor_Project/AutoML4SeaIce/preprocessed/' + name)
    # values = ds_train['SIC'].values
    # cond = values == 11  # in general, a boolean expression is required
    # values[cond] = float('NaN')
    # plotmap_ESA2(values)

    # name = "20180607T184226"
    # file = open("/home/svenvanc/Universiteit_Leiden/Bachelor_Project/AutoML4SeaIce/" + name + ".txt", "r+")
    # predarr = np.loadtxt(file)

    # name = '20181212T205512-y.npy'
    # ds_train = xr.open_dataset('/home/svenvanc/Universiteit_Leiden/Bachelor_Project/AutoML4SeaIce/preprocessed/' + name)
    # values = ds_train['SIC'].values
    # cond = values == 11  # in general, a boolean expression is required
    # values[cond] = float('NaN')
    # plotmap_ESA2(values)

    name = '20181212T205512-y.npy'
    y = np.load(name)
    y2 = np.load(name)
    namex = '20181212T205512-nersc-x.npy'
    x = np.load(namex)

    cond = x == 0.0
    cond = cond[:, :, 0]

    cond2 = x == 0.0
    cond2 = cond2[:, :, 1]

    y[cond] = 110.0
    y2[cond2] = 110.0

    plot_scene(None, y, y2, None)


