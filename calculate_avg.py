
"""Train function for models. Data is loaded using ASIP2Dataset. Validated using validate.py."""
"""

This one is using
* distribution_strategy=tf.distribute.MirroredStrategy(),
* multiple CPU / GPU
* batch size 32

Result:
* from 1050 to 915 sec per epoch  (not that much)
* giving an error: AUTO sharding policy will apply DATA sharding policy as it failed to apply FILE sharding polic
* see file:  kerasTunerTest-03_1444688.err

============ compared to -02
* loading all patches from 1 file. To avoid disk congestion, just to test
* using flat_map
"""


# -- Third-party modules -- #
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import r2_score, mean_squared_error

# --Proprietary modules -- #
from functions.data_functions import get_scene_lists
from routines.loaders import ASIP2Dataset, InfDataset
from init import OPTIONS


def remove_land_values(y_actual):
    y_actual = [item for sublist in y_actual for item in sublist]  # flatten the map dimensions (the first two dimensions)

    return list(filter(lambda a: a != 11, y_actual))            # takes 1 second


def calculate_mean(options):
    dataset = InfDataset(files=options['train_list'], options=options)

    batch_means = []
    for i, (x, y, mask, name, flip) in enumerate(tqdm(iterable=dataset, total=options['epoch_len'], colour='red')):
        print("\nBatch " + str(i))

        size = tf.size(y).numpy()
        distribution = []
        for j in range(0, 12):
            percentage = round((len(y[y == j]) / size) * 100, 1)
            distribution.append(percentage)
        print("Distribution: " + str(distribution))

        y = y[y != 11]
        batch_means.append(np.mean(y))

        print("Batch " + str(i) + " mean is " + str(round(batch_means[i], 1)))

    return np.mean(batch_means)


def predict(options, prediction):
    dataset = InfDataset(files=options['validate_list'], options=options)

    # Validate mean
    r2_values, rounded_r2_values, rmse_values, rounded_rmse_values, accuracy_values = [], [], [], [], []
    distribution = [[0 for x in range(0, 12)] for y in range(options['epoch_len'])]

    for i, (x, y, mask, name, flip) in enumerate(tqdm(iterable=dataset, total=len(options['validate_list']), colour='red')):
        print("\nBatch " + str(i))
        y = y[0].numpy()
        size = tf.size(y).numpy()

        for j in range(0, 12):
            distribution_array = np.full((len(y), len(y[0])), j)
            percentage_j = np.sum(y == distribution_array)
            distribution[i][j] = round((percentage_j / size), 3)

        y = remove_land_values(y)

        prediction_array = np.full((len(y)), prediction)
        rounded_prediction_array = np.full((len(y)), round(prediction))

        r2_i = r2_score(y, prediction_array)
        r2_values.append(r2_i)
        print("r2 value is: " + str(r2_i))

        rounded_r2_i = r2_score(y, rounded_prediction_array)
        rounded_r2_values.append(rounded_r2_i)
        print("rounded r2 value is: " + str(rounded_r2_i))

        rmse_i = np.sqrt(mean_squared_error(y, prediction_array))
        rmse_values.append(rmse_i)
        print("rmse value is: " + str(rmse_i))

        rounded_rmse_i = np.sqrt(mean_squared_error(y, rounded_prediction_array))
        rounded_rmse_values.append(rounded_rmse_i)
        print("rounded rmse value is: " + str(rounded_rmse_i))

        predictions = np.sum(y == rounded_prediction_array)
        accuracy_i = predictions / size
        accuracy_values.append(accuracy_i)
        print("accuracy value is: " + str(accuracy_i))

        print("distribution: " + str(distribution[i]))

    return np.mean(r2_values), np.mean(rounded_r2_values), np.mean(rmse_values), np.mean(rounded_rmse_values), \
           np.mean(accuracy_values), distribution


def main(options: dict):
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
    # Get training and validation scene list
    options['train_list'], options['validate_list'] = get_scene_lists()

    mean = calculate_mean(options)
    print("\n\n*************************")
    print("Mean prediction is: " + str(mean))
    print("*************************\n\n")

    r2, rounded_r2, rmse, rounded_rmse, accuracy, distribution = predict(options, mean)

    print("\n\n*************************")
    print("*       STATISTICS      *")
    print("*************************")

    print("\n Mean prediction is: " + str(mean))
    print(" - r2: " + str(r2))
    print(" - rounded_r2: " + str(rounded_r2))
    print(" - rmse: " + str(rmse))
    print(" - rounded_rmse: " + str(rounded_rmse))
    print(" - accuracy: " + str(accuracy))

    print("\n distribution: ")
    distribution = np.sum(distribution, axis=0)
    distribution[:] = [x / len(options['validate_list']) for x in distribution]
    print(distribution)


if __name__ == '__main__':
    main(OPTIONS)