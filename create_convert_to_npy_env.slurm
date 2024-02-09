#!/bin/bash
#SBATCH --job-name=hvm-env-xarray
#SBATCH --output=log/%x_%j.out
#SBATCH --error=log/%x_%j.err
#SBATCH --mem=1G
#SBATCH --time=00:30:00
#SBATCH --partition=testing
#SBATCH --ntasks=1


echo "# xarray install"
cd /home/s2358093/data1/hvm/hvm-alice

export ENV=/home/s2358093/data1/conda_envs/xarray

echo "[$SHELL] #### Starting Python test"
echo "[$SHELL] ## This is $SLURM_JOB_USER and this job has the ID $SLURM_JOB_ID"
# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] ## current working directory: "$CWD

conda info

# load modules (assuming you start from the default environment)
# we explicitly call the modules to improve reproducibility
# in case the default settings change
# conda env remove -n AutoML4SeaIce
#conda env remove -p ${ENV}
#echo "[$SHELL] ## ***** removed *****"


conda create --prefix ${ENV} python=3.10
echo "[$SHELL] ## ***** created *****"


__conda_setup="$('/cm/shared/easybuild/GenuineIntel/software/Miniconda3/4.9.2/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/cm/shared/easybuild/GenuineIntel/software/Miniconda3/4.9.2/etc/profile.d/conda.sh" ]; then
        . "/cm/shared/easybuild/GenuineIntel/software/Miniconda3/4.9.2/etc/profile.d/conda.sh"
    else
        export PATH="/cm/shared/easybuild/GenuineIntel/software/Miniconda3/4.9.2/bin:$PATH"
    fi
fi
unset __conda_setup


# conda activate AutoML4SeaIce
conda activate ${ENV}
echo "[$SHELL] ## ***** conda env activated *****"

conda info

pip install numpy xarray==2022.10.0 netCDF4~=1.5.8 -q

echo "conda prefix: $CONDA_PREFIX"


python --version


echo "[$SHELL] #### Finished Python test. Have a nice day"
