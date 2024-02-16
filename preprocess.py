"""
Function to preprocess ASIP2 scenes.
"""

# -- File info -- #
__author__ = 'Andreas R. Stokholm'
__contributors__ = 'Andrzej S. Kucik'
__copyright__ = ['Technical University of Denmark', 'European Space Agency']
__contact__ = ['stokholm@space.dtu.dk', 'andrzej.kucik@esa.int']
__version__ = '0.2.0'
__date__ = '2022-02-08'

# -- Built-in modules -- #
import glob
import os

# -- Third-part modules -- #
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# --Proprietary modules -- #
from functions.preprocessing_functions import PreprocessParallel
from functions.utils import colour_str, SCENE_VARIABLES
from init import OPTIONS

# Parameters
NUM_WORKERS = 4


def collate_function(x):
    return x


def main():
    """Run preprocessing routine."""
    print(colour_str('\n# -- Preprocessing Scenes -- #\n', 'cyan'))

    # List of .nc files
    if not os.path.exists(OPTIONS['path_to_processed_data']):
        os.mkdir(OPTIONS['path_to_processed_data'])

    dirlist = glob.glob(os.path.join(OPTIONS['path_to_data'], '*.nc'))

    # - Ignore files which were already processed
    dirlist = [file for file in dirlist if
               os.path.split(file)[-1][:15] + '_pro.nc' not in os.listdir(OPTIONS['path_to_processed_data'])
               or
               os.path.split(file)[-1][:15] + '_bins.npy' not in os.listdir('misc/scene_pro_bins')]

    if not len(dirlist):
        print('No scenes to process.')

    preprocess_fast = PreprocessParallel(files=dirlist,
                                         train_variables=SCENE_VARIABLES[:5],
                                         train_fill_value=OPTIONS['train_fill_value'],
                                         class_fill_values=OPTIONS['class_fill_values'],
                                         pixel_spacing=OPTIONS['pixel_spacing'],
                                         n_classes=OPTIONS['n_classes'])

    run_preprocess = DataLoader(preprocess_fast,
                                batch_size=None,
                                num_workers=NUM_WORKERS,
                                shuffle=False,
                                collate_fn=collate_function)

    # Process the scenes
    for scene, bins in tqdm(iterable=run_preprocess, total=len(dirlist), colour='green'):
        # - Check if the scenes' extreme values lie within the desired normalized range

        for variable in SCENE_VARIABLES[:5]:
            scene_min = np.min(scene[variable].values)
            scene_max = np.max(scene[variable].values)

            if scene_min < OPTIONS['normalize_range'][0]:
                print('Minimum value of:', colour_str(scene_min, 'blue'), 'in scene',
                      colour_str(scene.attrs['scene_id'], 'purple'), 'for variable ', colour_str(variable, 'green'))

            elif scene_max > OPTIONS['normalize_range'][1]:
                print('Maximum value of:', colour_str(scene_min, 'blue'), 'in scene',
                      colour_str(scene.attrs['scene_id'], 'purple'), 'for variable ', colour_str(variable, 'green'))

            elif np.isnan(np.sum(scene[variable])):
                print('NaN in', colour_str(scene.scene_id, 'purple'))

        # - Save the processed scene and the bins
        scene.to_netcdf(os.path.join(OPTIONS['path_to_processed_data'], scene.attrs['scene_id']))
        np.save('misc/scene_pro_bins/' + scene.attrs['scene_id'][:15] + '_bins', bins, allow_pickle=True)


if __name__ == '__main__':
    main()
