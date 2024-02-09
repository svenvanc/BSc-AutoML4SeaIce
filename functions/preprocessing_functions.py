#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Functions to be used for preprocessing ASIP2 scenes."""

# -- File info -- #
__author__ = 'Andreas R. Stokholm'
__contributors__ = 'Andrzej S. Kucik'
__copyright__ = ['Technical University of Denmark', 'European Space Agency']
__contact__ = ['stokholm@space.dtu.dk', 'andrzej.kucik@esa.int']
__version__ = '0.4.0'
__date__ = '2022-02-08'

# -- Built-in modules -- #
import os

# -- Third-party modules -- #
import numpy as np
import xarray as xr
# import torch
# from torch.utils.data import Dataset

# -- Proprietary modules -- #
from functions.utils import convert_polygon_icechart, CHARTS
from functions.visualization_functions import rotate_scene


class PreprocessParallel():
    """
    Parallel preprocessing of ASIP2 scenes. Returns scenes ready for saving.

    Note: for some reason the data loader unpacks the xarray dataset... quick fix:
        scene, scene_id = out[1:][0]
        scene_id = scene_id[0:15]
        to_save = xr.Dataset(scene)
    """

    def __init__(self,
                 files: list,
                 train_variables: list,
                 train_fill_value: int,
                 class_fill_values: list,
                 n_classes: list,
                 pixel_spacing: int = 40,
                 new_minmax: tuple = (-1, 1)
                 ):
        self.files = files
        self.train_variables = train_variables
        self.train_fill_value = train_fill_value
        self.class_fill_values = class_fill_values
        self.pixel_spacing = pixel_spacing
        self.new_minmax = new_minmax
        self.n_classes = n_classes

    def __len__(self):
        """Returns number of files. Required by pytorch."""

        return len(self.files)

    def __getitem__(self, idx):
        """
        Process scenes in parallel.

        Parameters
        ----------
        idx : int
            File index.

        Returns
        -------
        scene : xarray Dataset
            Scene from ASIP2.
        scene.attrs : xarray Dataset attributes
            PyTorch DataLoader unpacks xarray Datasets and removes attrs.
            This allows the reconstructions of the xarray Dataset.
        bins : ndarray
            1D numpy array with class count of ice charts.

        """
        scene, bins = preprocess_scenes_single(scene_path=self.files[idx],
                                               train_fill_value=self.train_fill_value,
                                               class_fill_values=self.class_fill_values,
                                               train_variables=self.train_variables,
                                               pixel_spacing=self.pixel_spacing,
                                               new_minmax=self.new_minmax,
                                               n_classes=self.n_classes)
        return scene, bins


def preprocess_scenes_single(scene_path: str,
                             train_fill_value: int,
                             class_fill_values: list,
                             train_variables: list,
                             n_classes: list,
                             pixel_spacing: int = 40,
                             new_minmax: tuple = (-1, 1),):
    """
    Preprocess single ASIP2 scene. Used in parallel computation together with pytorch data loader.

    Parameters
    ----------
    scene_path : str
        Path of the specific scene desired.
    train_fill_value : int
        Value to replace nan values in scene training data.
    class_fill_values : List[int]
        Values to replace nan values in scene ice charts, i.e. SIC, SOD, floe.
    train_variables : List
        Desired variables in scene
    n_classes: dict
        Number of classes for SIC, SOD, FLOE.
    pixel_spacing : int
        Pixel spacing of processed scenes. Original 40. Should be 40^N, e.g. 40, 80, 160 etc. > 40 downsamples scene.
    new_minmax: tuple
        New minimum and maximum value to normalize around.

    Returns
    -------
    new_xr :
        New xarray Dataset with train_variables. Other variables are not included.
    bins :
        Number of pixels belonging to each class.
    """
    scene = xr.open_dataset(scene_path)
    scene = convert_polygon_icechart(scene, train_bool=True)
    get_2d_incidenceangle(scene)
    if pixel_spacing > 40:
        scene = shrink_scene(scene, os.path.split(scene_path)[-1], train_variables, pixel_spacing=pixel_spacing)

    align_masks(scene, train_variables)
    scene = remove_excess(scene, train_variables)
    normalize_scene(scene, train_variables, new_minmax)
    replace_nan_scene(scene, train_variables, train_fill_value=train_fill_value, class_fill_values=class_fill_values)

    bins = get_scene_bins(scene, n_classes)

    return scene, bins


def align_masks(scene, train_variables: list):
    """
    Aligns masks (nan) in scene for train_variables and polygon_icechart. Variables altered directly
    in scene.

    Parameters
    ----------
    scene :
        xarray dataset; scenes from ASIP2.
    train_variables : List
        Desired variables in scene.
    """
    mask = np.isnan(scene['SIC'].values)

    # Check for nan in any variable. combined mask for all variables.
    for variable in train_variables:
        mask = np.logical_or(mask, np.isnan(scene[variable].values))

    # Apply mask in charts
    for chart in CHARTS:
        scene[chart].values[mask] = np.nan

    # Apply mask in train_variables
    for variable in train_variables:
        scene[variable].values[mask] = np.nan


def get_2d_incidenceangle(scene):
    """
    Replace the 1d sar_incidenceangles in ASIP2 scenes with a 2d version.

    Values are replaced directly in the xarray.

    Parameters
    ----------
    scene :
        xarray dataset; scenes from ASIP2.
    """
    # np.tile repeats 1d vector to match sar_primary scene shape
    new_xr = xr.DataArray(name='sar_incidenceangles',
                          data=np.tile(scene['sar_incidenceangles'].values, (scene['sar_primary'].values.shape[0], 1)),
                          dims=scene['sar_primary'].dims,
                          attrs=scene['sar_incidenceangles'].attrs)

    scene['sar_incidenceangles'] = new_xr


def get_scene_bins(scene, n_classes: list):
    """
        Return bins from processed scene.

        Parameters
        ----------
        scene :
            xarray dataset; scenes from ASIP2.
        n_classes : dict
            Number of classes for SIC, SOD, FLOE.

        Returns
        -------
        bincount : dict
            Dictionary of count of the CHARTS bins of the scene polygon_icechart.
        """
    bincount = {}
    for chart in CHARTS:
        bincount[chart] = np.bincount(np.ravel(scene[chart].values).astype(int), minlength=n_classes[chart])

    return bincount


def normalize(values, channels_minmax, new_min: float, new_max: float):
    """
    Normalize to new_min, new_max.

    Parameters
    ----------
    values :
        ndarray; values to normalize.
    channels_minmax :
        ndarray; minimum and maximum of channels.
    new_min : float
        Normalization minimum.
    new_max : float
        Normalization maximum.

    Returns
    -------
    values :
        Normalized numpy array.
    """
    values = (new_max - new_min) * (values - channels_minmax[0]) / (channels_minmax[1] - channels_minmax[0]) + new_min

    return values


def normalize_scene(scene, train_variables: list, new_minmax):
    """
    Normalize to +- 1 for full scene for train_variables.

    Parameters
    ----------
    scene :
        xarray dataset; scenes from ASIP2.
    train_variables : List
        Desired variables in scene
    new_minmax :
        list or 1D numpy array; values to normalize to, e.g. [-1,+1].
    """
    global_minmax = np.load('misc/global_minmax.npy', allow_pickle=True).item()
    # noinspection PyUnresolvedReferences
    scene_minmax = np.stack([global_minmax[variable] for variable in train_variables])

    for variable in train_variables:
        scene[variable].values = normalize(scene[variable].values,
                                           channels_minmax=scene_minmax[train_variables.index(variable), :],
                                           new_min=new_minmax[0],
                                           new_max=new_minmax[1])


def remove_excess(scene, train_variables: list):
    """
    Remove excess nan from all train_variables including polygon_icechart.

    Parameters
    ----------
    scene :
        xarray dataset; scenes from ASIP2.
    train_variables : List
        Desired variables in scene.

    Returns
    -------
    scene :
        xarray Dataset.
    """
    # new dims name in xarray
    dim_0 = 'final_dim_0'
    dim_1 = 'final_dim_1'

    sic_mask = np.isnan(scene['SIC'].values)
    # Remove rows and cols if all nan.
    for chart in CHARTS:
        tmp = scene[chart].values
        tmp_clean = tmp[:, ~sic_mask.all(axis=0)]
        tmp_sic_mask = sic_mask[:, ~sic_mask.all(axis=0)]
        tmp_clean = tmp_clean[~tmp_sic_mask.all(axis=1), :]

        # Replace old charts.
        tmp_attrs = scene[chart].attrs
        scene = scene.drop_vars(chart)
        scene = scene.assign({chart: xr.DataArray(tmp_clean, dims=[dim_0, dim_1], attrs=tmp_attrs)})

    # repeat for train_variables
    for variable in train_variables:
        tmp = scene[variable].values
        tmp_clean = tmp[:, ~np.isnan(tmp).all(axis=0)]
        tmp_clean = tmp_clean[~np.isnan(tmp_clean).all(axis=1), :]

        # Replace all images.
        tmp_attrs = scene[variable].attrs
        scene = scene.drop_vars(variable)
        scene = scene.assign({variable: xr.DataArray(data=tmp_clean, dims=[dim_0, dim_1], attrs=tmp_attrs)})

    return scene


def replace_nan_scene(scene, train_variables: list, train_fill_value: int, class_fill_values: list):
    """
    Replace nans in netCDF files with value for all variables including poly_icechart.

    Parameters
    ----------
    scene :
        xarray Dataset; scenes from ASIP2
    train_variables : List
        Desired variables in scene
    train_fill_value : int
        Value to fill in nans for training data.
    class_fill_values : List[int]
        List of values to fill in nans for reference data. [SIC, SOD, FLOE]
    """
    for idx, chart in enumerate(CHARTS):
        where_nan = np.isnan(scene[chart].values)
        scene[chart].values[where_nan] = class_fill_values[chart]

    for variable in train_variables:
        scene[variable].values[where_nan] = train_fill_value


# noinspection PyUnresolvedReferences
def shrink_scene(scene, scene_name, train_variables: list, pixel_spacing: int):
    """
    Reduce scene variable size by row, col: 1/2, 1/2.

    Parameters
    ----------
    scene : xarray Dataset
        Scenes from ASIP2.
    scene_name : str
        Name of scene.
    train_variables : List
        Desired variables in scene.
    pixel_spacing : int
        Pixel spacing of processed scenes. Original 40. Should be 40^N, e.g. 40, 80, 160 etc. > 40 downsamples scene.

    Returns
    -------
    new_xr : xarray Dataset
        new xarray Dataset with train_variables. Other variables are not included
    """
    stride = kernel_size = pixel_spacing // 40

    # Get flip integer for np.flip.
    flip = rotate_scene(scene)

    # Reduce charts' dimensions by with max pool stride 2
    tmp_data = torch.nn.functional.max_pool2d(torch.from_numpy(scene['SIC'].values).unsqueeze(0),
                                              kernel_size=kernel_size, stride=stride).squeeze(0).numpy()

    new_xr = xr.DataArray(name='SIC', data=tmp_data, attrs=scene['SIC'].attrs)
    for chart in CHARTS[1:]:
        tmp_data = torch.nn.functional.max_pool2d(torch.from_numpy(scene[chart].values).unsqueeze(0),
                                                  kernel_size=kernel_size, stride=stride).squeeze(0).numpy()
        new_xr = xr.merge([new_xr, xr.DataArray(name=chart, data=tmp_data, attrs=scene[chart].attrs)])

    # Reduce train_variables by with avg pool stride 2 and create new xarray Dataset
    for variable in train_variables:
        tmp_data = torch.nn.functional.avg_pool2d(
            torch.from_numpy(scene[variable].values).unsqueeze(0),
            kernel_size=kernel_size, stride=stride).squeeze(0).numpy()
        new_xr = xr.merge([new_xr, xr.DataArray(name=variable, data=tmp_data, attrs=scene[variable].attrs)])

    new_xr.attrs = ({'scene_id': scene_name[:15] + '_pro.nc',
                     'original_id': scene_name,
                     'flip': flip,
                     'latlon_dim': scene['lat'].values.shape,
                     'lat': scene['lat'].values.ravel(),
                     'lon': scene['lon'].values.ravel(),
                     })

    return new_xr


def random_crop(scene, train_variables, chart, crop_shape: tuple, fill_value: int):
    """
    Perform random cropping in scene with 'crop_shape'.
    Note: Does not have equal representation for pixels at edges.

    Parameters
    ----------
    scene :
        Xarray dataset; a scene from ASIP2.
    train_variables : dict
        Desired variables in scene.
    chart : str
        Desired chart parameter in scene.
    crop_shape : tuple
        (patch_height, patch_width)
    fill_value : int
        Value to pad in case of scene smaller than crop_shape.

    Returns
    -------
    patch :
        Numpy array with shape (len(train_variables), patch_height, patch_width).
    """
    scene_shape = (scene.dims['final_dim_0'], scene.dims['final_dim_1'])
    row_diff = crop_shape[0] - scene_shape[0]
    col_diff = crop_shape[1] - scene_shape[1]
    variables = np.hstack((chart, train_variables))

    # Check if scene smaller than crop_shape. If True, pad to enable crop.
    if (row_diff > 0) or (col_diff > 0):
        # Calculate number to pad to center the scene.
        # Clip negative numbers to 0 if scene larger than crop_shape.
        # Plus one so low != high in np.random.randint.
        row_pad = int(np.ceil(np.clip(row_diff, a_min=0, a_max=None) / 2) + 1)
        col_pad = int(np.ceil(np.clip(col_diff, a_min=0, a_max=None) / 2) + 1)

        scene = scene[variables].pad(final_dim_0=(row_pad, row_pad),
                                     final_dim_1=(col_pad, col_pad),
                                     constant_values=fill_value)

        # Update scene_shape.
        scene_shape = (scene.dims['final_dim_0'], scene.dims['final_dim_1'])

    row_rand = np.random.randint(low=0, high=scene_shape[0] - crop_shape[0])
    col_rand = np.random.randint(low=0, high=scene_shape[1] - crop_shape[1])

    patch = scene[variables].sel(
        final_dim_0=range(row_rand, row_rand + crop_shape[0]),
        final_dim_1=range(col_rand, col_rand + crop_shape[1])).to_array().values

    return patch
