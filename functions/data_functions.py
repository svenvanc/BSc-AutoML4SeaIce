#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This module will create the scenes_minmax array with minimum and maximum values, histograms and
preview of the AI4Arctic netCDF files.
"""

# -- File info -- #
__author__ = 'Andreas R. Stokholm'
__contributors__ = 'Andrzej S. Kucik'
__copyright__ = ['Technical University of Denmark', 'European Space Agency']
__contact__ = ['stokholm@space.dtu.dk', 'andrzej.kucik@esa.int']
__version__ = '0.3.0'
__date__ = '2022-02-08'

# -- Built-in modules -- #
import os

# -- Third-party modules -- #
import numpy as np
# import torch
import xarray as xr
from tqdm import tqdm

# -- Proprietary modules -- #
from functions.utils import check_nc_files, colour_str


def get_all_scenes_minmax(scene_list: list, path_dir: str, variables: list):
    """
    Get minmax of all variables for each scene in scene_list.

    Parameters
    ----------
    scene_list : List[str]
        List of scene names to get minmax from.
    path_dir : str
        Path to directory with scenes.
    variables : List
        List of variables to get minmax from.

    Returns
    -------
    ndarray of all scene minmax.
    """
    scene_list = check_nc_files(scene_list)

    scenes_n = np.size(scene_list)
    scenes_minmax = []

    for i in tqdm(range(scenes_n), position=0, leave=True):
        scene = xr.open_dataset(os.path.join(path_dir, scene_list[i]))
        scenes_minmax.append(get_scene_minmax(scene, variables))

    return np.stack(scenes_minmax, 0)


def get_global_minmax(scene_list: list, path_dir: str, variables: list):
    """
    Get global minmax of all scene variables in scene_list.

    Parameters
    ----------
    scene_list : List[str]
        List of scene names to get minmax from.
    path_dir : str
        Path to directory with scenes.
    variables : List
        List of variables to get minmax from.

    Returns
    -------
    global_minmax : dict
        Dict with global minmax from scenes.
    """
    # Gets minmax of all scenes in scene_list.
    scenes_minmax = get_all_scenes_minmax(scene_list, path_dir, variables)

    # Finds min max of each variable.
    global_minmax = np.stack([[np.nanmin(scenes_minmax[:, variables.index(variable), 0]),
                               np.nanmax(scenes_minmax[:, variables.index(variable), 1])]
                              for variable in variables], 0)

    # Converts numpy array to dict with array containing min max for each variable.
    global_minmax = {variable: global_minmax[variables.index(variable)] for variable in variables}

    return global_minmax


def get_scene_lists(scene_name_orig: bool = False):
    """
    Get training and validation train lists. Removes duplicates and invalid scenes.

    Parameters
    ----------
    scene_name_orig : bool, optional
        Should the scene name have original names or _pro.nc. The default is False.

    Returns
    -------
    train_list : List[str]
        List of scene names used for training.
    validate_list : List[str]
        List of scene names used for validation.
    """
    # List of the training and validate scenes
    # train_list = np.loadtxt('./datalists/train_list.txt', dtype=str)
    train_list = np.loadtxt('./datalists/train_alice.txt', dtype=str)
    # validate_list = set(np.loadtxt('./datalists/validate_list.txt', dtype=str))
    validate_list = set(np.loadtxt('./datalists/validate_alice.txt', dtype=str))
    # - Add extra validation scenes
    if 'validate_list_extra.txt' in os.listdir('./datalists'):
        validate_list |= set(np.loadtxt('./datalists/validate_list_extra.txt', dtype=str))
        validate_list = list(validate_list)

    # [:15] in the file name corresponds to YYYYMMDDThhmmss, which is the Sentinel-1 acquisition.
    # timestamp. The files have been named YYYYMMDDThhmmss_pro - pro implying processed.
    suffix = '_pro.nc'
    if scene_name_orig is False:
        train_list = [file[:15] + suffix for file in train_list]
        validate_list = [file[:15] + suffix for file in validate_list]

    # List of the invalid scenes
    if 'invalid_list.txt' in os.listdir('./datalists'):
        invalid_list = np.loadtxt('./datalists/invalid_list.txt', dtype=str)

        if scene_name_orig is False:
            invalid_list = [file[:15] + suffix for file in invalid_list]

        # Compares scenes in invalid_list and train_list, discards invalid_scenes
        train_list = [file for file in train_list if file not in invalid_list]

    # Select scenes in train_list not occurring in validate_list
    train_list = [file for file in train_list if file not in validate_list]
    train_list.sort()
    validate_list.sort()

    return train_list, validate_list


def get_scene_lists_supervised_gan(train_list: list, validate_list: list, supervised_fraction: int,
                                   scene_name_orig: bool = False):
    """
    Get supervised training lists for semi-supervised GAN.

    Parameters
    ----------
    train_list : list[str]
        Original list of training scenes.
    validate_list : list[str]
        Validation list to check for duplicates.
    supervised_fraction : int
        Percentage fraction of supervised scenes. 10, 25, 50 are implemented.
    scene_name_orig : bool, optional
        Should the scene name have original names or _pro.nc. The default is False.

    Returns
    -------
    train_list : List[str]
        List of scene names used for training.
    supervised_list : List[str]
        List of scene names used for semi-supervised GAN training.
    """
    supervised_list = set(np.loadtxt('./datalists/supervised_list.txt', dtype=str))
    supervised_list_extra = set(np.loadtxt('./datalists/supervised_list_extra.txt', dtype=str))

    # Get correct scene suffix.
    if scene_name_orig is False:
        suffix = '_pro.nc'
        train_list = [file[:15] + suffix for file in train_list]
        supervised_list = set([file[:15] + suffix for file in supervised_list])
        supervised_list_extra = set([file[:15] + suffix for file in supervised_list_extra])

    # Remove supervised scenes from train_list.
    train_list = [file for file in train_list if file not in supervised_list]

    # Check for duplicates in validate_list.
    supervised_vs_val = [file for file in supervised_list if file in validate_list]
    supervised_vs_val = supervised_vs_val + [file for file in supervised_list_extra if file in validate_list]

    if len(supervised_vs_val) > 0:
        print('supervised and validate lists are overlapping.')
        print(supervised_vs_val)

    # Get supervised list for 25 and 50 percentage fractions.
    if supervised_fraction == 25:
        supervised_list = set.union(supervised_list, supervised_list_extra)
        train_list = [file for file in train_list if file not in supervised_list_extra]

    elif supervised_fraction == 50:
        supervised_list = set.union(set(train_list[0::3]), supervised_list, supervised_list_extra)
        train_list = [file for file in train_list if file not in supervised_list]

    # In case supervised_fraction is entered incorrectly.
    elif supervised_fraction != 10:
        print('Supervised Fraction not equal to 10, 25 or 50! Aborting.')
        exit()

    return train_list, supervised_list


def get_scene_minmax(scene, variables: list):
    """
    Get minmax of variables in scene.

    Parameters
    ----------
    scene :
        xarray dataset; scenes from ASIP2.
    variables : List
        Scene variables to get the minmax of.

    Returns
    -------
    ndarray of scene minmax of variables.
    """
    scene_minmax = ([[scene[variable].attrs['min'], scene[variable].attrs['max']] for variable in variables[:4]])

    scene_minmax.append([np.min(scene[variables[4]].values), np.max(scene[variables[4]].values)])
    scene_minmax.append([np.min(scene[variables[5]].values), np.max(scene[variables[5]].values)])

    scene_minmax += [[scene[variable].attrs['min'], scene[variable].attrs['max']] for variable in variables[6:]]

    return np.stack(scene_minmax, 0).astype(float)


def get_scene_sampling_frequency(options: dict):
    """
    Get the number of times a scene in train_list is sampled to get similar pixel representation.

    Sampling scheme is based on the number of real value pixels in an image.
    The scheme will ensure that the minimal number of pixels in the training scenes
    is only sampled once, but scenes with e.g. 5 times the pixels, is sampled 5 times
    as often.

    Parameters
    ----------
    options : dict
        Dictionary with options for the training environment.

    Returns
    -------
    scene_sampling : 1D numpy array
        Array to random sampling scenes from.
    """
    print(colour_str('\n# -- Getting scene sampling weights -- #', 'cyan'))

    # Find number of real pixels in scenes (i.e. real numbers not nans).
    pixel_list = []
    for scene_name in tqdm(options['train_list'], colour='green'):
        scene = xr.open_dataset(os.path.join(options['path_to_processed_data'], scene_name))
        pixel_list.append(
            scene['SIC'].values[scene['SIC'].values < options['class_fill_values']['SIC']].size)

    # Find how often a scene should be sampled to have relatively similar representation.
    min_pixels = np.min(pixel_list)
    scene_frequency = [pixels / min_pixels for pixels in pixel_list]
    scene_prob = np.array(scene_frequency) / np.sum(scene_frequency)

    return list(scene_prob)


def get_bins_weight(options: dict):
    """
    Get class weights from bins of scenes in train_list located in path_bins.

    Weights are based on the median frequency scheme calculated as:
        w_class = class_median / class_total_samples

    Parameters
    ----------
    options : dict
        Dictionary with options for the training environment.

    Returns
    -------
    bins_w :
        ndarray weights for classes.
    """
    if options['loss_weighted']:
        bins_total = np.zeros(options['n_classes'][options['chart']])
        bins_w = np.zeros(options['n_classes'][options['chart']])

        for scene_name in tqdm(options['train_list'], colour='blue'):
            scene_bins = np.load(os.path.join('./misc', 'scene_pro_bins', scene_name[:15] + '_bins.npy'),
                                 allow_pickle=True).item()
            bins_total += scene_bins[options['chart']]

        bins_total[options['class_fill_values'][options['chart']]] *= options['class_fill_weight']

        bins_w = np.true_divide(bins_total.sum(), (options['n_classes'][options['chart']] - 1) * bins_total,
                                out=bins_w, where=bins_total != 0)

        # bins_w = torch.from_numpy(bins_w).to(torch.float32) #Todo off
        bins_w /= bins_w.sum()

        with np.printoptions(precision=8, suppress=True):
            print('Weight bins:\n{}'.format(bins_w.numpy()), end='\n')

    elif not options['loss_weighted']:
        bins_w = None
        # bins_w = torch.ones(options['n_classes'][options['chart']]) #Todo off
        # bins_w[options['class_fill_values'][options['chart']]] = 0
        print('Weight bins is', colour_str(None, 'orange'))

    else:
        bins_w = None

    print('\n')

    return bins_w


def labels_to_one_hot(options: dict, labels):
    """
    Convert labels to one-hot-encoded format.
    Parameters
    ----------
    options : dict
        Dictionary with options for the training environment.
    labels :
        ndTensor, true examples with dimensions (batch, height, width).

    Returns
    -------
    labels
        One-hot-encoded with dimensions (batch, n_classes, height, width).
    """
    # (batch, height, # width, n_classes)
    # labels = torch.nn.functional.one_hot(labels, #Todo off
    #                                      num_classes=options['n_classes'][options['chart']])

    # (batch, n_classes, height, width)
    # labels = labels.permute(0, 3, 1, 2).type(torch.float)  #Todo off

    labels = None
    return labels
