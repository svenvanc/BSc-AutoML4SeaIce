#!/bin/bash
#SBATCH --job-name=kerasTunerTest-r2_large_res
#SBATCH --output=log/%x_%j.out
#SBATCH --error=log/%x_%j.err
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:05:00
#SBATCH --partition=cpu-short
#SBATCH --ntasks=1


echo "With env hvm-05"

#cd /home/s2358093/AutoML4SeaIce
#cd /home/s2358093/data1/hvm/hvm-alice

export ENV=/home/s2358093/data1/conda_envs/hvm-05

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


conda activate ${ENV}
echo "[$SHELL] ## ***** conda env activated *****"

echo "conda prefix: $CONDA_PREFIX"

TENSORRT_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/tensorrt
echo "TENSORRT_PATH: $TENSORRT_PATH"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/:$TENSORRT_PATH/
#https://stackoverflow.com/questions/74956134/could-not-load-dynamic-library-libnvinfer-so-7
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

conda info
# one time needed
#pip install python-dotenv
#pip install keras-tuner -q


echo "=================== [$SHELL] Run python prog ========================="
python kerasTuner-20-get-results.py
echo "[$SHELL] Script finished"

