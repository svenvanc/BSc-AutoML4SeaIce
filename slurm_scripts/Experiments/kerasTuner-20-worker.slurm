#!/bin/bash
#SBATCH --job-name=kerasTuner_worker
#SBATCH --output=0_worker_lots_log/%x_%a_%j.out
#SBATCH --error=0_worker_lots_log/%x_%a_%j.err
#SBATCH --mem-per-cpu=10G
#SBATCH --time=11:59:00
#SBATCH --partition=gpu-medium
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:2
#SBATCH --array=1-101%3

# use this to run 100 workers, max 15 simultaneous: #SBATCH --array=1-100%15

cd /home/s2358093/AutoML4SeaIce
export ENV=/home/s2358093/data1/conda_envs/hvm-05

echo "[$SHELL] #### Starting Python test"
echo "[$SHELL] ## This is $SLURM_JOB_USER and this job has the ID $SLURM_JOB_ID"
echo "[$SHELL] ## This is part of an array job $SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID"

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
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

conda info

echo "parent_job_id:  ${PARENT_JOB_ID}"

source TUNER_ADDRESS_${PARENT_JOB_ID}.sh
echo "Will work with chief on ${KERASTUNER_ORACLE_IP}, on port nr ${KERASTUNER_ORACLE_PORT}"

export KERASTUNER_TUNER_ID="node-$SLURM_JOB_ID"
export KERASTUNER_ONE_TRIAL_WORKER_MODE=1

# Test if we can connect
nc -vz $KERASTUNER_ORACLE_IP $KERASTUNER_ORACLE_PORT

echo "Tuner_ID: $KERASTUNER_TUNER_ID"

sleep 10
python kerasTuner-20-a.py
echo "[$SHELL] Script finished"

