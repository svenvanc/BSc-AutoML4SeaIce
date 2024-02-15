"""Train function for models. Data is loaded using ASIP2Dataset. Validated using validate.py."""

# -- File info -- #
__author__ = ['Andreas R. Stokholm', 'Andrzej S. Kucik']
__copyright__ = ['Technical University of Denmark', 'European Space Agency']
__contact__ = ['stokholm@space.dtu.dk', 'andrzej.kucik@esa.int']
__version__ = '0.1.6'
__date__ = '2021-12-03'

# -- Built-in modules -- #
import os

# -- Third-party modules -- #
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

# --Proprietary modules -- #
from functions.loss_functions import get_scene_lists
from functions.loss_functions import calculate_loss
from functions.model_functions import get_optimizer, prep_model_dir, model_checkpoint, save_stats
from functions.utils import colour_str, print_options,
from routines.loaders import ASIP2Dataset
from routines.validate import net_validation


def train_net(net, device, bins_w, options: dict):
    """
    Model train function. Setup optimizer, dataset, dataloader, trains and validates model.

    Parameters
    ----------
    bins_w :
        Tensor with the weight bins.
    net :
        PyTorch model; Convolutional Neural Network to train.
    device :
        torch device: GPU or CPU used for computation.
    options : dict
        Dictionary with options for the training environment.
    """
    # Setup Optimizer, loss function and scheduler.
    # - Get training and validation scene list
    options['train_list'], options['validate_list'] = get_scene_lists()
    optimizer = get_optimizer(options, net)
    loss_avg = 0  # Initial loss average.

    # Prepare new model directory and training information file.
    model_dir = prep_model_dir(options=options)

    # Create a summary writer for tensorboard
    writer = SummaryWriter(log_dir=os.path.join(model_dir, './runs'))

    # Print training options and model directory.
    print_options(options=options, net=net)

    # Dataset
    dataset = ASIP2Dataset(files=options['train_list'], options=options)
    dataloader = DataLoader(dataset, batch_size=None, shuffle=True, num_workers=options['num_workers'], pin_memory=True)
    bins_w = bins_w.to(device)
    bins_w_sum = bins_w.sum()

    # -- Training Loop -- #
    for epoch in trange(options['epochs'], position=0):
        tqdm.write(f" epoch: {epoch + 1} / {options['epochs']}")
        torch.cuda.empty_cache()

        prev_loss_avg = loss_avg
        loss_cumu = torch.tensor([0.])
        net.train()

        # Loops though batches in queue.
        for i, (batch_x, batch_y) in enumerate(tqdm(iterable=dataloader, total=options['epoch_len'], colour='red')):
            # - Transfer to device.
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device)

            # - Forward pass.
            output = net(batch_x)

            # - Calculate loss for
            loss = calculate_loss(options=options, output=output, target=batch_y, bins_w=bins_w, bins_w_sum=bins_w_sum)

            # - Reset gradients from previous pass.
            optimizer.zero_grad()

            # - Backward pass.
            loss.backward()

            # - Optimizer step
            optimizer.step()

            # - Add batch loss.
            loss_cumu += loss.item()

            # - Average loss for displaying
            loss_avg = torch.true_divide(loss_cumu, i + 1).item()
            print('\rMean training loss: ' + colour_str(f'{loss_avg:.6f}', 'red'), end='\r')

        if epoch < options['epoch_before_val']:
            continue

        loss_delta = (loss_avg - prev_loss_avg)
        delta_colour = 'red' if loss_delta > 0 else 'green'
        print('Mean training loss:', colour_str(f'{loss_avg:.6f}', 'red'),
              '\tchange:', colour_str(f'{loss_delta:.6f}', delta_colour),
              '\tlr:', colour_str(str(optimizer.param_groups[0]['lr']), 'blue'), '\n')

        val_loss, acc_list, r2_list, all_acc, all_r2 = net_validation(options=options,
                                                                      net=net,
                                                                      device=device,
                                                                      model_dir=model_dir,
                                                                      epoch=epoch,
                                                                      bins_w=bins_w,
                                                                      bins_w_sum=bins_w_sum,
                                                                      test=False)

        # Save model performance
        save_stats(options=options,
                   optimizer=optimizer,
                   epoch=epoch,
                   train_loss=np.round(loss_avg, options['round_dec']),
                   val_loss=val_loss,
                   acc_list=acc_list,
                   r2_list=r2_list,
                   all_acc=all_acc,
                   all_r2=all_r2)

        # Get model checkpoint.
        model_checkpoint(model_dir=model_dir, net=net, optimizer=optimizer, epoch=epoch)

        # Save logs for tensorboard
        writer.add_scalars(main_tag='loss', tag_scalar_dict={'train': loss_avg, 'val': val_loss},
                           global_step=epoch)
        writer.add_scalars(main_tag='acc',
                           tag_scalar_dict={**{'all': all_acc}, **dict(zip(options['validate_list'], acc_list))},
                           global_step=epoch)
        writer.add_scalars(main_tag='r2',
                           tag_scalar_dict={**{'all': all_r2}, **dict(zip(options['validate_list'], r2_list))},
                           global_step=epoch)
        writer.add_scalar(tag='learning_rate/lr', scalar_value=optimizer.param_groups[0]['lr'], global_step=epoch)

    writer.flush()
    writer.close()
