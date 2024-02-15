"""Visualization functions."""

# -- File info -- #
__author__ = 'Andreas R. Stokholm'
__contributors__ = 'Andrzej S. Kucik'
__copyright__ = ['Technical University of Denmark', 'European Space Agency']
__contact__ = ['stokholm@space.dtu.dk', 'andrzej.kucik@esa.int']
__version__ = '0.2.4'
__date__ = '2021-11-04'

# -- Built-in modules -- #
import os

# -- Third-party modules -- #
import numpy as np
from matplotlib import pyplot as plt

# -- Proprietary modules -- #
from functions.utils import MONTHS, REGIONS

SIC_STRING = 'Sea Ice Concentration [%]'


def rotate_scene(scene):
    """
    Identify np.flip int [0,1] to flip scene correctly.

    Parameters
    ----------
    scene :
        xarray dataset; scenes from ASIP2.

    Returns
    -------
    flip : int
        Integer to be used in np.flip.
    """
    flip = 0

    # Check if lat is descending or ascending
    d_lat = scene['lat'].values[0, 0] - scene['lat'].values[-1, 0]
    # Check if lon is descending or ascending
    d_lon = abs(scene['lat'].values[0, 0]) - abs(scene['lat'].values[0, -1])

    if d_lat < 0:
        flip = 0
    elif d_lon < 0:
        flip = 1
    else:
        print('Other rotation, flip put to 0. Check rotation.')

    return flip


def scenes_distribution(scene_list: list, chart: str, path_bins: str):
    """
    Get scene distributions with respect to icechart classes, yearly monthly, and regional counts.

    Parameters
    ----------
    scene_list : List[str].
        List of scenes.
    path_bins : str
        Path to scene bins.

    Returns
    -------
    scenes_bins : ndarray
        1D Numpy array of bins of all scenes in scene_list.
    scenes_years: list of ndarrays
        List of 1D numpy arrays with yearly seasonal occurrences.
    reg_count : list
        List of regional scene counts.

    """
    scenes_dates = []
    scenes_regions = []
    scenes_bins = []

    for scene_name in scene_list:
        scenes_dates.append(np.array((int(scene_name[:4]), int(scene_name[4:6]))))
        scenes_regions.append(np.array(scene_name[45:-3]))
        scenes_bins.append(np.load(os.path.join(path_bins, scene_name[:15] + '_bins.npy'),
                                   allow_pickle=True).item()[chart])

    scenes_dates = np.stack(scenes_dates, axis=0)
    scenes_regions = np.stack(scenes_regions, axis=0)
    scenes_bins = np.stack(scenes_bins, axis=0)
    scenes_bins = np.sum(scenes_bins, axis=0)
    scenes_bins = scenes_bins[:-1]  # Remove masked class

    index_2018 = np.where(scenes_dates[:, 0] == 2018)[0]
    index_2019 = np.where(scenes_dates[:, 0] == 2019)[0]
    scenes_years = [scenes_dates[index_2018, 1], scenes_dates[index_2019, 1], scenes_dates[:, 1]]

    reg_count = [np.sum(np.char.equal(scenes_regions, region)) for region in REGIONS]

    return scenes_bins, scenes_years, reg_count


# noinspection PyUnresolvedReferences
def plot_scenes_distribution(options: dict, train_list: list, validate_list: list, path_bins: str):
    """
    Get regional and monthly distribution of scenes in scene_list.

    Parameters
    ----------
    options : dict
        Options for the model training and visualization environment.
    train_list : List[str].
        List of training scenes.
    validate_list : List[str].
        List of validation scenes.
    path_bins : str
        Path to scene bins.
    """
    rwidth = 0.8

    train_bins, train_dates, train_reg = scenes_distribution(scene_list=train_list, chart=options['chart'],
                                                             path_bins=path_bins)
    val_bins, val_dates, val_reg = scenes_distribution(scene_list=validate_list, chart=options['chart'],
                                                       path_bins=path_bins)

    # Seasonal histogram
    hist_dates = plt.figure()
    plt.grid(zorder=0)
    plt.hist([train_dates[2], val_dates[2]], bins=np.arange(1, 14) - .5, histtype='bar', rwidth=rwidth, zorder=3,
             color=[options['primary_color'], options['secondary_color']])
    plt.xticks(np.arange(1, 13), MONTHS, rotation=25, horizontalalignment='right')
    plt.ylabel('Number of Scenes')
    plt.legend(labels=['Train', 'Test'])

    # Regional histograms
    hist_reg = plt.figure()
    plt.grid(zorder=0)
    plt.bar(np.arange(0, len(REGIONS)) - 0.15, train_reg, width=.3, zorder=3, color=options['primary_color'])
    plt.bar(np.arange(0, len(REGIONS)) + 0.15, val_reg, width=.3, zorder=3, color=options['secondary_color'])
    plt.xticks(np.arange(0, len(REGIONS)), REGIONS, rotation=25,
               horizontalalignment='right')
    plt.xlabel('Regions')
    plt.ylabel('Number of Scenes')
    plt.legend(labels=['Train', 'Test'])

    # Bins histogram
    hist_bins = plt.figure()
    plt.grid(zorder=0)
    plt.bar(np.arange(0, train_bins.size) - 0.15, train_bins / train_bins.sum() * 100,
            width=.3, zorder=3, color=options['primary_color'])
    plt.bar(np.arange(0, train_bins.size) + 0.15, val_bins / val_bins.sum() * 100,
            width=.3, zorder=3, color=options['secondary_color'])
    plt.xticks(np.arange(0, train_bins.size), np.arange(0, train_bins.size),
               horizontalalignment='right')
    plt.xlabel(SIC_STRING[:-3] + '[Class #]')
    plt.ylabel('Percentage of Pixels [%]')
    plt.legend(labels=['Train', 'Test'])

    print('Percentage split -- train and validation --')
    print(f"Train: {train_bins.sum() / (train_bins.sum() + val_bins.sum())}")
    print(f"Validation: {val_bins.sum() / (train_bins.sum() + val_bins.sum())}")

    return hist_dates, hist_reg, hist_bins

