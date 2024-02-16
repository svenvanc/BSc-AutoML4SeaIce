#!/bin/bash
#SBATCH --job-name=hvm-env05
#SBATCH --output=log/%x_%j.out
#SBATCH --error=log/%x_%j.err
#SBATCH --mem=1G
#SBATCH --time=01:00:00
#SBATCH --partition=gpu-short
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1


echo # Build with tf 2.9.2"
cd /home/s2358093/data1/hvm/hvm-alice

export ENV=/home/s2358093/data1/conda_envs/hvm-05

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
conda env remove -p ${ENV}
echo "[$SHELL] ## ***** removed *****"


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

echo "set LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/

conda info

#See here for versions needed https://www.tensorflow.org/install/pip
conda install --prefix ${ENV} -c conda-forge cudatoolkit=11.2 cudnn=8.1.0

pip install nvidia-tensorrt==8.4.1.5
# create symbolic links
TENSORRT_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/tensorrt
ln -s ${TENSORRT_PATH}/libnvinfer_plugin.so.8 ${TENSORRT_PATH}/libnvinfer_plugin.so.7
ln -s ${TENSORRT_PATH}/libnvinfer.so.8 ${TENSORRT_PATH}/libnvinfer.so.7
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${TENSORRT_PATH}/

pip install tensorflow==2.9.2
#https://stackoverflow.com/questions/74956134/could-not-load-dynamic-library-libnvinfer-so-7
#https://github.com/tensorflow/tensorflow/issues/57679#issuecomment-1249197802
#See: https://github.com/google-research/multinerf/issues/47

#conda install --prefix ${ENV} matplotlib
pip install matplotlib
pip install keras-tuner -q
pip install python-dotenv

nvidia-smi

# echo "[$SHELL] ## before setting"
# export LD_LIBRARY_PATH=/home/s2358093/data1/conda_envs/testenv/lib:$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
# echo "[$SHELL] ## after setting"

echo "conda prefix: $CONDA_PREFIX"


python --version

PY_PROG=$(cat <<EOF
import tensorflow as tf
print('## TF version:')
print(tf. __version__)

print('## CPU setup:')
print(tf.reduce_sum(tf.random.normal([1000, 1000])))

print('## GPU setup physical:')
print(tf.config.list_physical_devices('GPU'))

print('## GPU setup logical:')
print(tf.config.list_logical_devices('GPU'))

print('## GPU Device name:')
print(tf.test.gpu_device_name())
EOF
)
python -c "$PY_PROG"

echo "[$SHELL] #### Finished Python test. Have a nice day"
