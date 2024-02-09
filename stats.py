import math
import os.path
import os
import scipy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from init import OPTIONS
from tensorflow import keras



# experiment_name = 'first_large_experiment'
# experiment_name = 'small_experiment'
# experiment_name = 'kerasTuner-20-a'
experiment_name = os.getenv('EXP_NAME')
input_type = os.getenv('INPUT_TYPE')
print("Python file, the experiment name is: ", experiment_name, ".")
print("Python file, the data type is: ", input_type, ".")

path_to_data = '/home/s2358093/data1/seaice_npy'
# path_to_image_storing = '/home/s2358093/data1/res_sven/images/' + experiment_name + '_128'
path_to_image_storing = '/home/s2358093/data1/res_sven/images/multiple_r2_experiments/' + experiment_name


# path_to_data = '/home/svenvanc/Universiteit_Leiden/Bachelor_Project/AutoML4SeaIce/'
# path_to_image_storing = '/home/svenvanc/Universiteit_Leiden/Bachelor_Project/AutoML4SeaIce/images'
# unet_model = keras.models.load_model('/home/svenvanc/Universiteit_Leiden/Bachelor_Project/AutoML4SeaIce/kerasTuner/kerasTuner-20-a')

# validate_names = [
#     '20181212T205512'
# ]

validate_names = [
    "20180315T184323",
    "20180327T201909",
    "20180521T183450",
    "20180607T184226",
    "20180610T195617",
    "20180619T184327",
    "20180626T183452",
    "20180627T205609",
    "20180720T120811",
    "20180730T180326",
    "20180804T114534",
    "20180903T082625",
    "20180909T100819",
    "20180910T081914",
    "20190208T081200",
    "20190330T102526",
    "20190409T104048",
    "20190423T102526",
    "20190509T081306",
    "20190517T102423",
    "20181212T205512",
]

# Names of the SIC classes.
SIC_GROUPS = {
    0: 0,
    1: 10,
    2: 20,
    3: 30,
    4: 40,
    5: 50,
    6: 60,
    7: 70,
    8: 80,
    9: 90,
    10: 100,
    11: 'unknown'  # Added
}

ICE_STRINGS = {
    'SIC': 'Sea Ice Concentration [%]',
    'SOD': 'Stage of Development',
    'FLOE': 'Floe Size'
}

# https://stackoverflow.com/questions/35751306/python-how-to-pad-numpy-array-with-zeros
def get_patch(x, y):
    # patch_size = 4000
    # patch_size = 1040
    # patch_size = 512
    n_rows, n_cols = y.shape
    # print(n_rows, n_cols)


    # rows_to_use = (n_rows // 256) * 256
    # cols_to_use = (n_cols // 256) * 256
    # x1 = x[:rows_to_use, :cols_to_use, :]
    # y1 = y[:rows_to_use, :cols_to_use]

    rows_to_add = 0
    cols_to_add = 0
    if (n_rows % 256) != 0:
        rows_to_add = (256 - (n_rows % 256))
    if (n_cols % 256) != 0:
        cols_to_add = (((n_cols // 256) + 1) * 256) - n_cols

    x1 = np.pad(x, [(0, rows_to_add), (0, cols_to_add), (0, 0)], mode='constant', constant_values=0)
    y1 = np.pad(y, [(0, rows_to_add), (0, cols_to_add)], mode='constant', constant_values=0)

    # x1 = x[row:row + patch_size, col:col + patch_size, :]
    # y1 = y[row:row + patch_size, col:col + patch_size]
    y1 = np.expand_dims(y1, axis=-1)
    return x1, y1


def get_prediction(model, x):
    predictions = model.predict(x)
    predictions = np.squeeze(predictions)
    predictions = np.argmax(predictions, axis=2)
    return predictions


def apply_unknown_mask_to_y(x, y):
    mask = x == 0.0
    mask = mask[:, :, :, 0:1]
    # mask = mask[:, :, :, 1:2]
    mask = np.squeeze(mask)
    y[mask] = 11


def apply_unknown_mask_to_y_to_255(x, y):
    mask = x == 0.0
    mask = mask[:, :, :, 0:1]
    mask = np.squeeze(mask)
    y[mask] = 255


def save_prediction(prediction, land_values_mask, name, column_size):
    prediction = prediction.astype(float)
    prediction[land_values_mask] = float('NaN')

    row = int(len(prediction) / column_size)
    col = column_size
    prediction_2D = np.reshape(prediction, (row, col))

    file = open("/home/s2358093/data1/tmp/" + name + ".txt", "w+")

    np.savetxt(file, prediction_2D)
    file.close()
    return


def plot_scene(x, y, prediction, name):
    fig, axs = plt.subplots(nrows=1, ncols=2)

    apply_unknown_mask_to_y(x, y)
    apply_unknown_mask_to_y(x, prediction)

    cmap = OPTIONS['cmap'][OPTIONS['chart']]
    cmap = cmap.with_extremes(over='white')  # class 11 ==> unknown

    axs[0].imshow(y, vmin=OPTIONS['vmin'][OPTIONS['chart']], vmax=OPTIONS['vmax'][OPTIONS['chart']],
                  cmap=cmap)
    axs[0].set_xticks([])
    axs[0].set_yticks([])

    axs[1].imshow(prediction, vmin=OPTIONS['vmin'][OPTIONS['chart']], vmax=OPTIONS['vmax'][OPTIONS['chart']],
                  cmap=cmap)
    axs[1].set_xticks([])
    axs[1].set_yticks([])

    plt.subplots_adjust(wspace=0.05)
    arranged = np.arange(0, len(SIC_GROUPS.values()))
    ncolor = OPTIONS['cmap'][OPTIONS['chart']].N
    norm = colors.BoundaryNorm(arranged - 0.5, ncolor)  # Get colour boundaries. -0.5 to center ticks for each color.

    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=OPTIONS['cmap'][OPTIONS['chart']]),
                      ticks=arranged,
                      location='bottom',
                      ax=axs,
                      pad=0.05,
                      # shrink=0.8,
                      fraction=0.06
                      )
    cb.set_label(label=ICE_STRINGS['SIC'])
    cb.set_ticklabels(list(SIC_GROUPS.values()))

    model_name = os.path.join(path_to_image_storing, name + '.png')
    fig.savefig(model_name,
                dpi=128,
                format='png',
                bbox_inches='tight',
                pad_inches=0.05,
                transparent=False)
    plt.close('all')


def calc_r2(actual, prediction, mean):
    # corr_matrix = np.corrcoef(actual, prediction)
    # corr = corr_matrix[0, 1]
    # R_sq = corr ** 2
    # print(R_sq, flush=True)

    # slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(actual, prediction)
    # R_sq = r_value ** 2
    # print(R_sq, flush=True)

    rss = np.sum((actual - prediction) ** 2)
    tss = np.sum((actual - mean) ** 2)

    test_mean = np.mean(actual)
    test_tss = np.sum((actual - test_mean) ** 2)
    R2 = 1 - (rss / test_tss)
    print("R2 custum func: ", R2, flush=True)

    return rss, tss


def generate_stats(model, file_name, mean):
    full_file_name = os.path.join(path_to_data, file_name)

    # TODO make sure its sar or nersc
    x = np.load(full_file_name + f'-{input_type}-x.npy')
    # x = np.load(full_file_name + '-nersc-x.npy')
    y = np.load(full_file_name + '-y.npy')

    x1, y1 = get_patch(x, y)
    x1 = np.expand_dims(x1, axis=0)

    predictions = get_prediction(model, x1)
    y1_copy = y1.copy()
    predictions_copy = predictions.copy()

    apply_unknown_mask_to_y(x1, predictions)
    apply_unknown_mask_to_y(x1, y1)

    y1_flattened = y1.flatten()
    predictions_flattened = predictions.flatten()
    print(predictions_flattened.shape, flush=True)


    # TODO can't mask with predictions (model predicts 11)
    predictions_flattened = predictions_flattened[y1_flattened != 11]
    y1_flattened = y1_flattened[y1_flattened != 11]
    print(predictions_flattened.shape, flush=True)

    n_equal = np.sum(predictions_flattened == y1_flattened)

    n_datapoints = len(y1_flattened)
    plot_scene(x1, y1_copy, predictions_copy, file_name)
    rss, tss = calc_r2(y1_flattened, predictions_flattened, mean)
    return (n_datapoints, rss, tss, n_equal)

total_value = 0
total_elements = 0
for file_name in validate_names:
    full_file_name = os.path.join(path_to_data, file_name)
    # TODO make sure its sar or nersc
    x = np.load(full_file_name + f'-{input_type}-x.npy')
    # x = np.load(full_file_name + '-nersc-x.npy')
    y = np.load(full_file_name + '-y.npy')

    x = np.expand_dims(x, axis=0)
    apply_unknown_mask_to_y(x, y)

    y_flattened = y.flatten()
    y_flattened = y_flattened[y_flattened != 11]
    print('\nElements: ', y_flattened.shape[0], flush=True)

    total_elements += y_flattened.shape[0]
    total_value += np.sum(y_flattened)

mean = total_value / total_elements                 # TODO should this be rounded?
print("The mean is: ", mean, flush=True)


def calc_model(model):

    if not os.path.exists(path_to_image_storing):
        os.makedirs(path_to_image_storing)

    rss_total = 0
    tss_total = 0
    n_datapoints_total = 0
    n_equal_total = 0
    for file_name in validate_names:
        print("\nWorking on file: " + file_name, flush=True)
        n_datapoints, rss, tss, n_equal = generate_stats(model, file_name, mean)
        rss_total += rss
        tss_total += tss
        n_datapoints_total += n_datapoints
        n_equal_total += n_equal

    acc = n_equal_total / n_datapoints_total
    mse = rss_total / n_datapoints_total
    rmse = math.sqrt(mse)
    r2 = 1 - (rss_total / tss_total)
    print("\nThe total accuracy score is: ", acc, flush=True)
    print("\nThe total RMSE score is: ", rmse, flush=True)
    print("\nThe total R2 score is: ", r2, flush=True)



unet_model = keras.models.load_model('/home/s2358093/data1/res_sven/model_results/multiple_r2_experiments/' + experiment_name, compile=False)
calc_model(unet_model)
