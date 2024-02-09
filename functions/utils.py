#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Utility functions."""

# -- File info -- #
__author__ = ['Andrzej S. Kucik', 'Andreas R. Stokholm']
__copyright__ = ['Technical University of Denmark', 'European Space Agency']
__contact__ = ['stokholm@space.dtu.dk', 'andrzej.kucik@esa.int']
__contact__ = ['stokholm@space.dtu.dk', 'andrzej.kucik@esa.int']
__version__ = '1.0.1'
__date__ = '2022-02-08'

# -- Built-in modules -- #
import copy
import csv
import os

# -- Third-party modules -- #
import numpy as np
# import torch
# import xarray as xr

# -- Proprietary modules -- #
# from models.unet import UNet #Todo off

# Colour dictionary
COLOURS = {'red': '\033[0;31m',
           'black': '\033[0m',
           'green': '\033[0;32m',
           'orange': '\033[0;33m',
           'purple': '\033[0;35m',
           'blue': '\033[0;34m',
           'cyan': '\033[0;36m'}

# Month list
MONTHS = ['January', 'February', 'March', 'April', 'May', 'June',
          'July', 'August', 'September', 'October', 'November', 'December']

# Region list
REGIONS = ['NorthWest', 'CentralWest', 'SouthWest', 'CapeFarewell', 'SouthEast',
           'NorthAndCentralEast', 'CentralEast', 'NorthEast', 'Qaanaaq']

CHARTS = ['SIC', 'SOD', 'FLOE']

# Variables in the ASIP2 data set
# noinspection SpellCheckingInspection
SCENE_VARIABLES = [
    # -- Sentinel-1 variables -- #
    'sar_primary',
    'sar_secondary',
    'nersc_sar_primary',
    'nersc_sar_secondary',
    'sar_incidenceangles',

    # -- AMSR2 channels -- #
    'btemp_6.9h', 'btemp_6.9v',
    'btemp_7.3h', 'btemp_7.3v',
    'btemp_10.7h', 'btemp_10.7v',
    'btemp_18.7h', 'btemp_18.7v',
    'btemp_23.8h', 'btemp_23.8v',
    'btemp_36.5h', 'btemp_36.5v',
    'btemp_89.0h', 'btemp_89.0v',

    # -- Misc variables -- #
    'distance_map',
]

# Sea Ice Concentration (SIC) code to class conversion lookup table.
SIC_LOOKUP = {
    'polygon_idx': 0,  # Index of polygon number.
    'total_sic_idx': 1,  # Total Sea Ice Concentration Index, CT.
    'sic_partial_idx': [2, 5, 8],  # Partial SIC polygon code index. CA, CB, CC.
    0: 0,
    1: 0,
    2: 0,
    10: 1,
    20: 2,
    30: 3,
    40: 4,
    50: 5,
    60: 6,
    70: 7,
    80: 8,
    90: 9,
    91: 10,
    92: 10,
    'mask': np.nan
}

# Stage of Development code to class conversion lookup table.
SOD_LOOKUP = {
    'sod_partial_idx': [3, 6, 9],  # Partial SIC polygon code index. SA, SB, SC.
    'threshold': 0.5,  # < 1. Minimum partial percentage SIC of total SIC to select SOD. Otherwise ambiguous polygon.
    'invalid': -9,  # Value for polygons where the SOD is ambiguous or not filled.
    'water': 0,
    0: 0,
    80: 0,
    81: 1,
    82: 1,
    83: 2,
    84: 2,
    85: 2,
    86: 4,
    87: 3,
    88: 3,
    89: 3,
    91: 4,
    93: 4,
    95: 5,
    96: 5,
    97: 5,
    98: 6,
    'mask': np.nan
}

# Ice floe/form code to class conversion lookup table.
FLOE_LOOKUP = {
    'floe_partial_idx': [4, 7, 10],  # Partial SIC polygon code index. FA, FB, FC.
    'threshold': 0.5,  # < 1. Minimum partial concentration to select floe. Otherwise polygon may be ambiguous.
    'invalid': -9,  # Value for polygons where the floe is ambiguous or not filled.
    'water': 0,
    0: 1,
    1: 1,
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 8,
    'fastice_class': 7,
    'mask': np.nan
}

ICECHART_NOT_FILLED_VALUE = -9


def get_dict_int_values(some_dic: dict):
    ints = [i for i in list(some_dic.values()) if type(i) == int]
    return ints


def check_nc_files(scene_list: list):
    """
    Remove non .nc files from scene list.

    Parameters
    ----------
    scene_list : List[str]
        List of scene names.

    Returns
    -------
    scene_list :
        ndarray of .nc only scene names
    """
    scene_list = np.array(scene_list)
    nc_files = np.core.defchararray.find(scene_list, '.nc') >= 1

    if np.any(nc_files) is False:
        print('Not all the files in directory are .nc files. Removed from loading list.')
        scene_list = scene_list[nc_files]

    return scene_list


def colour_str(word, colour: str):
    """Function to colour strings."""
    return COLOURS[colour.lower()] + str(word) + COLOURS['black']


def convert_polygon_icechart(scene, train_bool: bool = False):
    """
    Original polygon_icechart in ASIP2 scenes consists of codes to a lookup table `polygon_codes`.

    This function looks up codes and converts them. There are two options, 1
    (train_bool == True) creates 12 discrete classes.
    (nan values are replaced by function 'replace_nan_scene')

    polygon_icechart is replaced directly in xarray Dataset.

    Parameters
    ----------
    scene :
        xarray dataset ;scenes from ASIP2.

    train_bool : bool
        Option selection.
    """
    # Get codes from polygon_codes.
    codes = np.stack(np.char.split(scene['polygon_codes'].values.astype(str),
                                   sep=';'), 0)[SIC_LOOKUP['total_sic_idx']:, :-1].astype(int)

    # Convert codes to classes for Total and Partial SIC.
    converted_codes = copy.deepcopy(codes)
    for key, value in SIC_LOOKUP.items():
        if type(key) == int:
            for partial_idx in SIC_LOOKUP['sic_partial_idx']:
                tmp = converted_codes[:, partial_idx]
                if key in tmp:
                    converted_codes[:, partial_idx][np.where((tmp == key))] = value

            tmp = converted_codes[:, SIC_LOOKUP['total_sic_idx']]
            if key in tmp:
                converted_codes[:, SIC_LOOKUP['total_sic_idx']][np.where((tmp == key))[0]] = value

    # Find where partial concentration is empty but total SIC exist.
    ice_ct_ca_empty = np.logical_and(
        converted_codes[:, SIC_LOOKUP['total_sic_idx']] > SIC_LOOKUP[0],
        converted_codes[:, SIC_LOOKUP['sic_partial_idx'][0]] == ICECHART_NOT_FILLED_VALUE)
    # Assign total SIC to partial concentration when empty.
    converted_codes[:, SIC_LOOKUP['sic_partial_idx'][0]][ice_ct_ca_empty] = \
            converted_codes[:, SIC_LOOKUP['total_sic_idx']][ice_ct_ca_empty]

    # Convert codes to classes for partial SOD.
    for key, value in SOD_LOOKUP.items():
        if type(key) == int:
            for partial_idx in SOD_LOOKUP['sod_partial_idx']:
                tmp = converted_codes[:, partial_idx]
                if key in tmp:
                    converted_codes[:, partial_idx][np.where((tmp == key))] = value

    # Convert codes to classes for partial FLOE.
    for key, value in FLOE_LOOKUP.items():
        if type(key) == int:
            for partial_idx in FLOE_LOOKUP['floe_partial_idx']:
                tmp = converted_codes[:, partial_idx]
                if key in tmp:
                    converted_codes[:, partial_idx][np.where((tmp == key))] = value

    # Get matching partial ice classes, SOD.
    sod_a_b_bool = converted_codes[:, SOD_LOOKUP['sod_partial_idx'][0]] == \
        converted_codes[:, SOD_LOOKUP['sod_partial_idx'][1]]
    sod_a_c_bool = converted_codes[:, SOD_LOOKUP['sod_partial_idx'][0]] == \
        converted_codes[:, SOD_LOOKUP['sod_partial_idx'][2]]
    sod_b_c_bool = converted_codes[:, SOD_LOOKUP['sod_partial_idx'][1]] == \
        converted_codes[:, SOD_LOOKUP['sod_partial_idx'][2]]

    # Get matching partial ice classes, FLOE.
    floe_a_b_bool = converted_codes[:, FLOE_LOOKUP['floe_partial_idx'][0]] == \
        converted_codes[:, FLOE_LOOKUP['floe_partial_idx'][1]]
    floe_a_c_bool = converted_codes[:, FLOE_LOOKUP['floe_partial_idx'][0]] == \
        converted_codes[:, FLOE_LOOKUP['floe_partial_idx'][2]]
    floe_b_c_bool = converted_codes[:, FLOE_LOOKUP['floe_partial_idx'][1]] == \
        converted_codes[:, FLOE_LOOKUP['floe_partial_idx'][2]]

    # Remove matches where SOD == -9 and FLOE == -9.
    sod_a_b_bool[np.where(converted_codes[:, SOD_LOOKUP['sod_partial_idx'][0]] == -9)] = False
    sod_a_c_bool[np.where(converted_codes[:, SOD_LOOKUP['sod_partial_idx'][0]] == -9)] = False
    sod_b_c_bool[np.where(converted_codes[:, SOD_LOOKUP['sod_partial_idx'][1]] == -9)] = False
    floe_a_b_bool[np.where(converted_codes[:, FLOE_LOOKUP['floe_partial_idx'][0]] == -9)] = False
    floe_a_c_bool[np.where(converted_codes[:, FLOE_LOOKUP['floe_partial_idx'][0]] == -9)] = False
    floe_b_c_bool[np.where(converted_codes[:, FLOE_LOOKUP['floe_partial_idx'][1]] == -9)] = False

    # Arrays to loop over to find locations where partial SIC will be combined for SOD and FLOE.
    sod_bool_list = [sod_a_b_bool, sod_a_c_bool, sod_b_c_bool]
    floe_bool_list = [floe_a_b_bool, floe_a_c_bool, floe_b_c_bool]
    compare_indexes = [[0, 1], [0, 2], [1,2]]

    # Arrays to store how much to add to partial SIC.
    sod_partial_add = np.zeros(converted_codes.shape)
    floe_partial_add = np.zeros(converted_codes.shape)

    # Loop to find
    for idx, (compare_idx, sod_bool, floe_bool) in enumerate(zip(compare_indexes, sod_bool_list, floe_bool_list)):
        tmp_sod_bool_indexes = np.where(sod_bool)[0]
        tmp_floe_bool_indexes = np.where(floe_bool)[0]
        if tmp_sod_bool_indexes.size:  #i.e. is array is not empty.
            sod_partial_add[tmp_sod_bool_indexes, SIC_LOOKUP['sic_partial_idx'][compare_idx[0]]] = \
                converted_codes[:, SIC_LOOKUP['sic_partial_idx'][compare_idx[1]]][tmp_sod_bool_indexes]

        if tmp_floe_bool_indexes.size:  # i.e. is array is not empty.
            floe_partial_add[tmp_floe_bool_indexes, SIC_LOOKUP['sic_partial_idx'][compare_idx[0]]] = \
                converted_codes[:, SIC_LOOKUP['sic_partial_idx'][compare_idx[1]]][tmp_floe_bool_indexes]

    # Create arrays for charts.
    scene_tmp = copy.deepcopy(scene['polygon_icechart'].values)
    sic = copy.deepcopy(scene['polygon_icechart'].values)
    sod = copy.deepcopy(scene['polygon_icechart'].values)
    floe = copy.deepcopy(scene['polygon_icechart'].values)

    # Add partial concentrations when classes have been merged in conversion (see SIC, SOD, FLOE tables).
    tmp_sod_added = converted_codes + sod_partial_add.astype(int)
    tmp_floe_added = converted_codes + floe_partial_add.astype(int)

    # Find and replace all codes with SIC, SOD and FLOE.
    with np.errstate(divide='ignore', invalid='ignore'):
        for i in range(codes.shape[0]):
            code_match = np.where(scene_tmp == converted_codes[i, SIC_LOOKUP['polygon_idx']])
            sic[code_match] = converted_codes[i, SIC_LOOKUP['total_sic_idx']]
            if np.divide(np.max(tmp_sod_added[i, SIC_LOOKUP['sic_partial_idx']]),
                    tmp_sod_added[i, SIC_LOOKUP['total_sic_idx']]) * 100 >= SOD_LOOKUP['threshold'] * 100:

                # Find dominant partial ice type.
                sod[code_match] = converted_codes[i, SOD_LOOKUP['sod_partial_idx']][
                    np.argmax(tmp_sod_added[i, SIC_LOOKUP['sic_partial_idx']])]
            else:
                sod[code_match] = ICECHART_NOT_FILLED_VALUE

            if np.divide(np.max(tmp_floe_added[i, SIC_LOOKUP['sic_partial_idx']]),
                    tmp_floe_added[i, SIC_LOOKUP['total_sic_idx']]) * 100 >= FLOE_LOOKUP['threshold'] * 100:
                floe[code_match] = converted_codes[i, FLOE_LOOKUP['floe_partial_idx']][
                    np.argmax(tmp_floe_added[i, SIC_LOOKUP['sic_partial_idx']])]
            else:
                floe[code_match] = ICECHART_NOT_FILLED_VALUE

            if any(converted_codes[i, FLOE_LOOKUP['floe_partial_idx']] == FLOE_LOOKUP['fastice_class']):
                floe[code_match] = FLOE_LOOKUP['fastice_class']

    # Add masked pixels for ambiguous polygons.
    sod[sod == SOD_LOOKUP['invalid']] = SOD_LOOKUP['mask']
    floe[floe == FLOE_LOOKUP['invalid']] = FLOE_LOOKUP['mask']

    # Ensure water is identical across charts.
    sod[sic == SIC_LOOKUP[0]] = SOD_LOOKUP['water']
    floe[sic == SIC_LOOKUP[0]] = FLOE_LOOKUP['water']

    # Add the new charts to scene and add descriptions:
    scene = scene.assign({'SIC': xr.DataArray(sic, dims=scene['polygon_icechart'].dims)})
    scene = scene.assign({'SOD': xr.DataArray(sod, dims=scene['polygon_icechart'].dims)})
    scene = scene.assign({'FLOE': xr.DataArray(floe, dims=scene['polygon_icechart'].dims)})
    scene['SIC'].attrs['polygons'] = 'Sea Ice Concentration (SIC)'
    scene['SOD'].attrs['polygons'] = 'Stage of Development (SOD)'
    scene['FLOE'].attrs['polygons'] = 'Floe size or Form of sea ice (FLOE)'

    return scene


def estimate_coefficient(x, y):
    """
    Estimate coefficients to convert from ASIP V2 SAR [-1:1] range to dB.

    Parameters
    ----------
    x :
        1D array, input range with steps .
    y :
        1D array, desired output range with steps

    Returns
    -------
    Coefficients for conversion.
    """
    # Number of observations/points
    n = np.size(x)

    # Mean of x and y vector
    m_x, m_y = np.mean(x), np.mean(y)

    # Calculating cross-deviation and deviation about x
    ss_xy = np.sum(y * x) - n * m_y * m_x
    ss_xx = np.sum(x * x) - n * m_x * m_x

    # Calculating regression coefficients
    b_1 = ss_xy / ss_xx
    b_0 = m_y - b_1 * m_x

    return b_0, b_1


def print_options(options, net=None):
    """
    Print program options.

    Parameters
    ----------
    options : dict
        Dictionary with options for the training environment.
    net :
        PyTorch CNN network. May be list in case of GAN.
    """

    model_dir = os.path.join(os.getcwd(), 'saved_models', options['model_name'])

    with open(os.path.join(model_dir, 'options.txt'), mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file, delimiter=':')
        for key, value in options.items():
            writer.writerow([key, value])

        if net is not None:
            net = net if isinstance(net, list) else [net]
            for n in net:
                writer.writerow(['net', n])


def get_gpu_resources(gpu_ids: list, net):
    """
    Allocate device to net and get GPU and cpu resources.

    Parameters
    ----------
    gpu_ids : List[int]
        List with the indices of GPU devices to be used
    net :
        PyTorch model.

    Returns
    -------
    net :
        PyTorch model.
    device : torch.device
        Main GPU device.
    """
    # GPU
    # if torch.cuda.is_available(): # Todo off
    #     print(colour_str('GPU available!', 'green'))
    #     print('Total number of available devices: ', colour_str(torch.cuda.device_count(), 'orange'))
    #     device = torch.device(f'cuda:{gpu_ids[0]}')
    #     net.to(device)
    # 
    #     # - Check how many GPUs are going to be used
    #     if len(gpu_ids) > 1:
    #         print('Running on', colour_str('multiple', 'orange'), 'GPUs:')
    #         net = torch.nn.DataParallel(net, device_ids=gpu_ids)
    #     elif len(gpu_ids) == 1:
    #         print('Running on a', colour_str('single', 'orange'), 'GPU:')
    # 
    #     # - Print GPU IDs and names
    #     for idx in gpu_ids:
    #         print('\t\tID:', colour_str(idx, 'blue') + ', name:', colour_str(torch.cuda.get_device_name(idx), 'purple'))
    # 
    # # CPU
    # else:
    #     print(colour_str('GPU devices not available. Running on the CPU.', 'red'))
    #     device = torch.device('cpu')
    #     net = net.to(device)
    # 
    # print('\n')

    net = None
    device = None
    return net, device



def get_nets(options: dict):
    """
    Get the network to train.
    Parameters
    ----------
    options : dict
        Options dictionary.

    Returns
    -------
    nets :
        A network to train, for `unet` and `resunet`, or a list of networks: [generator_xy, generator_yx,
        discriminator_x, discriminator_y] for `gan`.
    """
    titles = {'unet': 'U-Net', 'resunet': 'Residual U-Net', 'gan': 'CycleGAN'}
    assert options['architecture'] in titles.keys(), colour_str('Architecture not yet implemented!', 'red')

    print('Using the', colour_str(titles[options['architecture']], 'purple'), 'architecture.\n')

    if options['architecture'] == 'unet':
        # return UNet(options=options) #Todo off
        return None