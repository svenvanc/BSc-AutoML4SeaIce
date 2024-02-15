
# -- Built-in modules -- #
import os
from os import listdir

# -- Third-party modules -- #
import numpy as np
import xarray as xr

# TODO new: dont hardcode paths
read_path = '/home/s2358093/data1/seaice_preprocessed'
write_path = '/home/s2358093/data1/seaice_npy'

SAR_VARIABLES = ['sar_primary', 'sar_secondary']
NERSC_VARIABLES = ['nersc_sar_primary', 'nersc_sar_secondary']


def get_all_files(mypath):
    from os.path import isfile, join
    return [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.startswith('20')]


def get_x_sar(scene):
    x_patch = scene[SAR_VARIABLES].to_array().values
    x_patch = np.moveaxis(x_patch, 0, -1)
    return x_patch


def get_x_nersc(scene):
    x_patch = scene[NERSC_VARIABLES].to_array().values
    x_patch = np.moveaxis(x_patch, 0, -1)
    return x_patch


def get_y(scene):
    chart = 'SIC'
    data = scene[chart].values
    return data.astype(np.uint8)


def convert(file_name: str):
    scene = xr.open_dataset(os.path.join(read_path, file_name))  # Open the scene.

    patch_x = get_x_nersc(scene)
    patch_y = get_y(scene)

    scene_name = file_name.removesuffix('_pro.nc')

    np.save(os.path.join(write_path, scene_name + '-nersc-x.npy'), patch_x)
    np.save(os.path.join(write_path, scene_name + '-y.npy'), patch_y)

    patch_x = get_x_sar(scene)
    np.save(os.path.join(write_path, scene_name + '-sar-x.npy'), patch_x)

    print('converted file', file_name)


r = get_all_files(read_path)

count = 0
for f in r:
    count += 1
    print('Converting: ', count, '...')
    convert(f)

print('Done!')
