

# About project
This project contains the code for the Mean prediction and Bayesian optimistation of sea ice concentration,
along with scripts to gather and visualise the results.


## Index
 - creating environments
   - preprocessing step one
   - preprocessing step two
   - Bayesian optimisation
 - Prerequisites
 - Downloading the data
 - Preprocessing
   - Step one
   - Step two
 - Experiments
   - Mean prediction
   - Bayesian optimisation
 - Plotting and statistics
   - Preprocessing results
   - Printing statistics
   - Plotting maps
 - Acknowledgements





## Create environments
This project uses different conda environments depending on the scripts executed.

- preprocessing step one:  
  This step uses [Python](https://www.python.org/) 3.9, and requires [PyTorch](https://www.pytorch.org/) 1.10.0 
library and its dependencies.
- preprocessing step two:  
  consists of the following files:
  - `create_convert_to_npy_env.slurm`
- Bayesian optimisation, plotting and statistics:  
  consists of the following files:
  - `create_bayesian_env.slurm`

## Prerequisite
- The `init.py` file is used by some scripts. Please configure the following variables:
  - `project_path`
  - `path_to_data` 
  - `path_to_processed_data`
  - `path_to_npy_data`
  - `path_to_results`
- The `.env` is used by the Bayesian optimisation process and the plotting and statistics scripts.
Please configure the following variables:
  - `PATH_TO_DATA`
  - `OUTPUT_DIR`
  - `INPUT_TYPE`
  - `EXP_NAME`
  




## Data download
Download the data from the following url:
https://data.dtu.dk/articles/dataset/AI4Arctic_ASIP_Sea_Ice_Dataset_-_version_2/13011134

Download this data in a directory and configure the `path_to_data` variable in the file `init.py` to this directory.


## Preprocessing
### Step one
Changing .nc to .nc:  
  Set the variable `path_to_processed_data`. This is where the resulting data will end up.
  Then run the script `preprocess.py` with `preprocess.slurm`. 

### Step two
Changing .nc to .npy:  
  Set the variable `path_to_npy_data`. This is where the resulting data will end up.
  Then run the script `convertDataToNpy.py` with `convert_to_npy.slurm`. 
  
Please configure the following variables in the .env file, as these are needed for later scripts:
  - `read_path`
  - `write_path`



## Experiments
### Mean prediction
The mean predicting script `calculate_avg.py` can be run with the `calculate_avg.slurm` slurm script.
The results of the experiment will be printed in the terminal.




### Bayesian optimisation
The bayesian optimisation scripts consists of the following files:  
  - kerasTuner-20-a.py  
  - kerasTuner-20-chief.slurm
  - kerasTuner-20-worker.slurm  

Both the chief process and the workers use the file `kerasTuner-20-a.py`. The chief process should be started first.
This can be done with the script `kerasTuner-20-chief.slurm`. 
This script will then automatically execute the `kerasTuner-20-worker.slurm` script.




## Plotting and statistics
### Processing results:
Before the results can be printed they must first be processed. This is done with the file `kerasTuner-20-get-results.py`
with its corresponding script `kerasTuner-20-get-resutls.slurm`. The following directory must be created: 
`/model_results` within the `OUTPUT_DIR` directory specified in the `.env` file.

### Printing statistics:
After the results have been processed the statistics can be read. This is done with the file `stats.py`
with its corresponding script `stats.slurm`. This script will print statistics and plot maps based on the predictions.
Before running the script set the `INPUT_TYPE` and `EXP_NAME` variables in the `.env` file.

### Plot maps:
with the file `plot_map.py` more custom plots can be made. This is to be used as a template to create your own custom plots. 
This file does not have a corresponding slurm script.

## Acknowledgements
* [Andreas Stokholm](https://github.com/astokholm/)
* [Andrzej Kucik](https://github.com/AndrzejKucik/)
* [DTU Space](https://www.space.dtu.dk/)
* [ESA &Phi;-Lab](https://philab.phi.esa.int/) 
