#!/bin/bash
#SBATCH --job-name=stats
#SBATCH --output=0_stats_real_log/%x_%j.out
#SBATCH --error=0_stats_real_log/%x_%j.err
#SBATCH --mem=10G
#SBATCH --time=0:30:00
#SBATCH --partition=cpu-short
#SBATCH --ntasks=1



# ENV=/home/s2358093/data1/conda_envs/tensorflow_10
# ENV=/home/s2358093/data1/conda_envs/tensorflow_10_conda
export ENV=/home/s2358093/data1/conda_envs/hvm-05

echo "This is experiment $EXP_NAME"
echo "This is experiment $INPUT_TYPE"
echo "[$SHELL] #### Starting Python test"
echo "[$SHELL] ## This is $SLURM_JOB_USER and this job has the ID $SLURM_JOB_ID"
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

conda activate $ENV
echo "[$SHELL] ## conda env activated"

python --version

export LD_LIBRARY_PATH=$ENV/lib:$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/


# Run the file
echo "[$SHELL] ## Run script"
python stats.py
echo "[$SHELL] ## test done"

echo "[$SHELL] #### Finished Python test. Have a nice day"