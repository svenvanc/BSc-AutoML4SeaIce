#!/bin/bash
#SBATCH --job-name=avg_land_short
#SBATCH --output=log_stats/%x_%j.out
#SBATCH --error=log_stats/%x_%j.err
#SBATCH --mem=50G
#SBATCH --time=2:00:00
#SBATCH --partition=cpu-short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1


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


conda activate AutoML4SeaIce
echo "conda env activated"

python --version

export LD_LIBRARY_PATH=/home/s2358093/.conda/envs/AutoML4SeaIce/lib:$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/


# Run the file
echo "[$SHELL] ## Run script"
python calculate_avg.py
echo "[$SHELL] #### Finished Python test. Have a nice day"