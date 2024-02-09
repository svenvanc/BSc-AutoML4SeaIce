#!/bin/bash
#SBATCH --job-name=convertDataToNpy-01
#SBATCH --output=log/%x_%j.out
#SBATCH --error=log/%x_%j.err
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --partition=testing
#SBATCH --ntasks=1

echo "With env convertDataToNpy-01"

cd /home/s2358093/data1/hvm/hvm-alice

export ENV=/home/s2358093/data1/conda_envs/xarray

echo "[$SHELL] #### Starting Python test"
echo "[$SHELL] ## This is $SLURM_JOB_USER and this job has the ID $SLURM_JOB_ID"
# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] ## current working directory: "$CWD


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

echo "conda prefix: $CONDA_PREFIX"

conda info


echo "[$SHELL] Run script"
python convertDataToNpy.py
echo "[$SHELL] Script finished"
