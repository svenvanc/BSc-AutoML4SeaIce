

# About project



## Index






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
  - `path_to_data` 
  - `path_to_processed_data`
- The `.env` is used by the Bayesian optimisation process and the plotting and statistics scripts.
Please configure the following variables:
  - `PATH_TO_DATA`
  - `OUTPUT_DIR`
  




## Data download
Download the data from the following url:
https://data.dtu.dk/articles/dataset/AI4Arctic_ASIP_Sea_Ice_Dataset_-_version_2/13011134

Download this data in a directory and configure the `path_to_data` variable in the file `init.py` to this directory.


## Preprocessing
### Step one
    - changing .nc to .nc

### Step two
consists of the following files:
  - convert_to_npy.slurm
  - convertDataToNpy.py
  - 
Please configure the following variables in the . file:
  - `read_path`
  - `write_path`



## Mean prediction





## Bayesian optimisation
    - chief
    - worker
    - .py


## Plotting and statistics
    - get-results
    - stats
    - plot map

## Acknowledgements
* [Andreas Stokholm](https://github.com/astokholm/)
* [Andrzej Kucik](https://github.com/AndrzejKucik/)
* [DTU Space](https://www.space.dtu.dk/)
* [ESA &Phi;-Lab](https://philab.phi.esa.int/) 
