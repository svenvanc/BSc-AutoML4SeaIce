#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Validate function for models. Data is loaded using InfDataset."""

# -- File info -- #
__author__ = 'Andreas R. Stokholm'
__contributors__ = 'Andrzej S. Kucik'
__copyright__ = ['Technical University of Denmark', 'European Space Agency']
__contact__ = ['stokholm@space.dtu.dk', 'andrzej.kucik@esa.int']
__version__ = '0.1.9'
__date__ = '2021-12-02'

# -- Built-in modules -- #
import os

# -- Third-party modules -- #
import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from sklearn.metrics import r2_score, confusion_matrix, plot_confusion_matrix
from torch.utils.data import DataLoader

# -- Proprietary modules -- #
from functions.data_functions import labels_to_one_hot
from functions.loss_functions import get_loss_function
from functions.validation_functions import calc_stats, visualize_gan_patches, print_validation_stats, val_final_layer
from functions.visualization_functions import show_pixel_count, show_chart_colorbar
from routines.loaders import InfDataset


def net_validation(options, net, device, model_dir, epoch, bins_w, test=False):
    """
    Validate model using scenes in options['validate_list'].
    Parameters
    ----------
    options : dict
        Dictionary with options for the training environment.
    net :
        PyTorch model; Convolutional Neural Network to train.
    device :
        Torch device: GPU or CPU used for computation.
    model_dir : str
        The directory of the trained model
    epoch : int
        Current training epoch.
    bins_w :
        Class weight bins.
    test : bool
        Whether the model is being tested or validated. True = test, with high dpi etc.
    Returns
    -------
    loss_avg : float
        Average validation loss.
    acc_list : List[float]
        Accuracy for each scene in validate_list.
    r2_list : List[float]
        R2 coefficients for each scene in validate_list.
    all_acc : float
        Accuracy for all validation pixels.
    all_r2 : float
        R2 for all validation pixels.
    """
    loss_function = get_loss_function(loss_name=options['loss'],
                                      bins_w=bins_w.to(device),
                                      ignore_index=options['class_fill_values'][options['chart']])

    dataset = InfDataset(options=options, files=options['validate_list'])
    asip_loader = DataLoader(dataset, batch_size=None, num_workers=options['num_workers'], shuffle=False)
    loss_cumu = torch.tensor([0.])  # Cumulative loss.
    matches_cumu = 0  # Cumulative number of matches.
    n_pixels = 0  # Cumulative number of pixels.
    r2_list = []  # Store R2 for each scene.
    acc_list = []  # Store accuracy for each scene.
    y_pixels = []  # Store label pixels.
    output_pixels = []  # Store prediction pixels.
    scene_n = 1  # Keep track of how many scenes has been validated.
    if test:
        dpi = 512
    else:
        dpi = 128

    # Inference loop.
    for inf_x, inf_y, mask, scene_name, flip in asip_loader:
        inf_x = inf_x.to(device)
        inf_y = inf_y.to(device)

        with torch.no_grad():
            net.eval()
            output = net(inf_x)

        # - Remove mask before calculating loss and stats.
        loss = options['loss_lambda'] * loss_function(input=output, target=inf_y)
        loss_cumu += loss.item()

        # - Get final categorical segmentation.
        output = val_final_layer(options=options, output=output)

        # - Remove excess axis need for model forward pass.
        output = output.cpu().squeeze()
        inf_y = inf_y.cpu().squeeze()
        output_flat = output[~mask]
        inf_y_flat = inf_y[~mask]

        # - Save inference images of selected scenes.
        if scene_name in options['scene_select']:
            output = output.numpy().astype(float)
            output[mask] = np.nan
            fig_output = plt.figure()
            output = np.flip(output, flip)
            plt.imshow(output, vmin=options['vmin'][options['chart']], vmax=options['vmax'][options['chart']],
                       cmap=options['cmap'][options['chart']])

            show_pixel_count(options)
            show_chart_colorbar(options, options['chart'])

            fig_output.savefig(os.path.join(model_dir, 'inference_' + scene_name[:15] + '_' + str(epoch) + '.png'),
                               dpi=dpi,
                               format='png',
                               bbox_inches='tight',
                               pad_inches=0,
                               transparent=True)
            plt.close('all')

        # - Calculate stats.
        r2, acc, matches_cumu, n_pixels = calc_stats(options=options, inf_y_flat=inf_y_flat,
                                                     pred_flat=output_flat, matches_cumu=matches_cumu,
                                                     n_pixels=n_pixels, scene_name=scene_name, scene_n=scene_n)
        r2_list.append(r2)
        acc_list.append(acc)
        y_pixels += list(inf_y_flat.numpy())
        output_pixels += list(output_flat.numpy())
        scene_n += 1

    # Validation stats.
    loss_avg = np.round(torch.true_divide(loss_cumu, np.size(options['validate_list'])).item(), options['round_dec'])
    all_acc = np.round((matches_cumu / n_pixels) * 100, options['round_dec'])
    all_r2 = np.round(r2_score(y_pixels, output_pixels) * 100, options['round_dec'])
    aux_acc = []

    if options['chart'] == 'SIC':
        # % of true intermediate predicted as intermediate
        y_pixels = np.array(y_pixels)
        output_pixels = np.array(output_pixels)
        matches_all = y_pixels == output_pixels
        true_water = y_pixels == 0
        water_acc = np.round(matches_all[true_water].sum() / true_water.sum() * 100, options['round_dec'])
        true_ice100 = y_pixels == options['n_classes']['SIC'] - 2
        ice100_acc = np.round(matches_all[true_ice100].sum() / true_ice100.sum() * 100, options['round_dec'])
        true_int = np.logical_and(y_pixels > 0, y_pixels < options['n_classes']['SIC'] - 2)
        pred_true_int = output_pixels[true_int]
        pred_int = np.logical_and(pred_true_int > 0, pred_true_int < options['n_classes']['SIC'] - 2)
        int_in_int = np.round(pred_int.sum() / true_int.sum() * 100, options['round_dec'])
        avg_macro_bin_acc = np.round(np.mean([water_acc, ice100_acc, int_in_int]), options['round_dec'])
        aux_acc = [water_acc, ice100_acc, int_in_int, avg_macro_bin_acc]

    # do something with the stats...

    return loss_avg, acc_list, r2_list, all_acc, all_r2, aux_acc

        # - Calculate and store stats.
        r2, acc, matches_cumu, n_pixels = calc_stats(options=options, inf_y_flat=labels_flat,
                                                     pred_flat=fake_y_flat, matches_cumu=matches_cumu,
                                                     n_pixels=n_pixels, scene_name=scene_name, scene_n=scene_n)
        r2_list.append(r2)
        acc_list.append(acc)
        y_pixels += list(labels_flat.numpy())
        pred_pixels += list(fake_y_flat.numpy())
        scene_n += 1

    # Validation stats.
    all_acc = np.round((matches_cumu / n_pixels) * 100, options['round_dec'])
    all_r2 = np.round(r2_score(y_pixels, pred_pixels) * 100, options['round_dec'])
    print_validation_stats(all_acc=all_acc, all_r2=all_r2, acc_list=acc_list, r2_list=r2_list)

    return acc_list, r2_list, all_acc, all_r2