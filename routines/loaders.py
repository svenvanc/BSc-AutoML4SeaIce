"""Pytorch Dataset class for training. Function used in train.py."""

# -- File info -- #
__author__ = 'Andreas R. Stokholm'
__contributors__ = 'Andrzej S. Kucik'
__copyright__ = ['Technical University of Denmark', 'European Space Agency']
__contact__ = ['stokholm@space.dtu.dk', 'andrzej.kucik@esa.int']
__version__ = '0.2.5'
__date__ = '2021-11-13'

# -- Built-in modules -- #
import os

# -- Third-party modules -- #
import imgaug.augmenters as iaa
import numpy as np
import xarray as xr
import tensorflow as tf

# -- Proprietary modules -- #
from functions.preprocessing_functions import random_crop


class ASIP2Dataset:
    """Pytorch dataset for loading batches of patches of scenes from the ASIP V2 data set."""

    def __init__(self, options, files):
        self.options = options
        self.files = files
        self.iterator_len = 0
        # self.scene = xr.open_dataset(os.path.join(self.options['path_to_processed_data'],self.files[0]))

        # Channel numbers in patches, includes reference channel.
        self.patch_c = self.options['train_variables'].size + 1

    def __len__(self):
        """
        Provide number of iterations per epoch. Function required by Pytorch dataset.

        Returns
        -------
        Number of iterations per epoch.
        """
        return self.options['epoch_len']

    # noinspection PyTypeChecker
    @staticmethod
    def prep_dataset(patches):
        """
        Convert patches from 4D numpy array to 4D torch tensor.

        Parameters
        ----------
        patches : ndarray
            Patches sampled from ASIP V2 scenes [PATCH, CHANNEL, H, W].

        Returns
        -------
        x :
            4D torch tensor; ready training data.
        y :
            3D torch tensor; reference data for training data x.
        """
        # Convert training data to tensor.
        x = tf.convert_to_tensor(patches[:, 1:])
        x = tf.cast(x, tf.float64)

        # Convert channel 0 (polygon icechart) to tensor.
        y = tf.convert_to_tensor(patches[:, 0])
        y = tf.cast(y, tf.int64)

        return x, y

    def data_augment(self, patches):
        """
        Augment data in patches. Random augments data in batches in no particular order.

        Parameters
        ----------
        patches : ndarray
            patches sampled from ASIP V2 scenes [batch size, channel, height, width].
            Not augmented at this stage.

        Returns
        -------
        patches : ndarray
            Augmented batch of patches. [batch size, channel, height, width].
        """
        seq = self.augmentation()

        # imgaug requires axis in order [channel, height, width, batch size].
        # [batch size, channel, height, width] -> [channel, height, width, batch size].
        patches = np.moveaxis(patches, 1, -1)
        patches = seq(images=patches)  # Augment data
        # [channel, height, width, batch size] -> [batch size, channel, height, width].
        patches = np.moveaxis(patches, -1, 1)

        if self.options['train_fill_value'] != self.options['class_fill_values'][self.options['chart']]:
            patches[:, 1:][patches[:, 1:] == self.options['class_fill_values'][self.options['chart']]] \
                = self.options['train_fill_value']

        return patches

    def augmentation(self):
        """
        Augmentation scheme.

        Provides 8 square symmetries by combining 4 90, degree rotations and 50 %
        change of flip vertically. In addition, between 1-4 geometric distortions are
        sometimes used depending on OPTIONS['sometimes'].

        Returns
        -------
        seq :
            imgaug sequence; scheme for data augmentation.
        """
        sometimes_flip = lambda aug: iaa.Sometimes(0.5, aug)
        sometimes = lambda aug: iaa.Sometimes(self.options['sometimes'], aug)
        seq = iaa.Sequential([
            # 8 square symmetries.
            iaa.Rot90([0, 1, 2, 3]),
            sometimes_flip(iaa.Flipud(1)),

            # Additional augmentation.
            sometimes(iaa.SomeOf((1, 4), [
                iaa.Affine(rotate=(-44.999, 44.999),
                           scale={'x': (1 - self.options['scale'], 1 + self.options['scale']),
                                  'y': (1 - self.options['scale'], 1 + self.options['scale'])},
                           shear=(-self.options['shear'], self.options['shear']),
                           translate_percent={
                               'x': (0 - self.options['translate'], 0 + self.options['translate']),
                               'y': (0 - self.options['translate'], 0 + self.options['translate'])},
                           order=0,
                           cval=self.options['class_fill_values'][self.options['chart']],
                           mode='constant')
            ]))
        ])

        return seq

    def __getitem__(self, idx):
        """
        Get batch. Function required by Pytorch dataset.

        Returns
        -------
        x :
            4D torch tensor; ready training data.
        y :
            If GAN: 4D torch tensor; Reference data for training data x.
            If CNN: 3D torch tensor; Reference data for training data x.
        mask :
            if GAN: 4D torch tensor; Mask from reference data.
        """
        if self.iterator_len >= self.options['epoch_len']:
            raise StopIteration
        self.iterator_len += 1

        # Placeholder to fill with data.
        patches = np.zeros((self.options['batch_size'], self.patch_c,
                            self.options['patch_size'], self.options['patch_size']))
        sample_n = 0

        # Continue until batch is full.
        while sample_n < self.options['batch_size']:
            # - Open memory location of scene. Uses 'Lazy Loading'.
            # - If sampling scheme is available in OPTIONS['sampling_weight'] use it.
            try:
                scene_id = self.options['sampling_weight'].multinomial(num_samples=1)
            except KeyError:
                scene_id = np.random.randint(low=0, high=len(self.files), size=(1,))[0]

            # - Load scene
            scene = xr.open_dataset(os.path.join(self.options['path_to_processed_data'],
                                                 self.files[scene_id]))
            # - Extract patches
            try:
                scene_patches = [random_crop(scene=scene,
                                             train_variables=self.options['train_variables'],
                                             chart=self.options['chart'],
                                             crop_shape=(self.options['patch_size'], self.options['patch_size']),
                                             fill_value=self.options['class_fill_values'][self.options['chart']])
                                 for _ in range(self.options['n_patches'])]
            except ValueError:
                print(f"{self.files[scene_id]} not large enough for patch")
                continue
            finally:
                scene.close()

            # - Discard patches with too many meaningless pixels (optional).
            if self.options['fill_threshold'] > 0:
                scene_patches = [patch for patch in scene_patches
                                 if np.sum(patch[0] != self.options['class_fill_values'][self.options['chart']])
                                 >= self.options['fill_threshold']]

                # - Only proceed if there are any meaningful patches
                num_patches = len(scene_patches)
                if num_patches > 0:
                    # -- Discard excessive patches and convert to a numpy array
                    scene_patches = np.array(scene_patches[:self.options['batch_size'] - sample_n])

                    # -- Stack the scene patches in patches
                    patches[sample_n:sample_n + len(scene_patches)] = scene_patches

                    # -- Update the index
                    sample_n += len(scene_patches)

        #TODO:
        # In the future more methods like data augmentation can be added. One of the ways to start implementing this
        # could be with the following sub method:
        # patches = self.data_augment(patches)

        # Prepare training arrays
        x, y = self.prep_dataset(patches)

        return x, y


class InfDataset:
    """Pytorch dataset for loading full scenes from the ASIP V2 dataset for inference."""

    def __init__(self, options, files):
        self.options = options
        self.files = files

    def __len__(self):
        """
        Provide number of iterations. Function required by Pytorch dataset.

        Returns
        -------
        Number of scenes per validation.
        """
        return len(self.files)

    def prep_scene(self, scene):
        """
        Convert patches from 4D numpy array to 4D torch tensor.

        Parameters
        ----------
        scene :

        Returns
        -------
        x :
            4D torch tensor, ready training data.
        y :
            3D torch tensor; reference data for training data x.
        mask :
            2D torch tensor; mask to remove nan values for displaying and calculating stats.
        """
        # Converts remaining data to tensor
        x = np.stack([scene[variable].values for variable in self.options['train_variables']], 0)
        x = tf.convert_to_tensor(x)
        x = tf.expand_dims(x, axis=0)

        # Converts channel 0 (polygon icechart) to tensor and scales ice concentration to 0-11.
        y = tf.convert_to_tensor(scene[self.options['chart']].values)
        y = tf.expand_dims(y, axis=0)
        y = tf.cast(y, tf.int64)

        mask = tf.convert_to_tensor(scene[self.options['chart']].values ==
                                    self.options['class_fill_values'][self.options['chart']])

        return x, y, mask

    def __getitem__(self, idx):
        """
        Get scene. Function required by Pytorch dataset.

        Returns
        -------
        x :
            4D torch tensor; ready inference data.
        y :
            3D torch tensor; reference inference data for x.
        mask :
            2D torch tensor; mask to remove nan values for loss calculation.
        name : str
            Name of scene.
        flip : int
            Integer to flip scene to display correctly. Used in np.flip(scene, flip)

        """
        scene = xr.open_dataset(os.path.join(self.options['path_to_processed_data'], self.files[idx]))
        flip = scene.attrs['flip']

        x, y, mask = self.prep_scene(scene)
        name = self.files[idx]

        scene.close()
        return x, y, mask, name, flip

