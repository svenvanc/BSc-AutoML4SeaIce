#!/bin/bash
#SBATCH --job-name=preprocessing2_testing
#SBATCH --output=log/%x_%j.out
#SBATCH --error=log/%x_%j.err
#SBATCH --mail-user="sven.van.c@gmail.com"
#SBATCH --mail-type="ALL"
#SBATCH --mem=50000M
#SBATCH --time=01:00:00
#SBATCH --partition=cpu-short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4



echo "[$SHELL] #### Starting Python test"
echo "[$SHELL] ## This is $SLURM_JOB_USER and this job has the ID $SLURM_JOB_ID"
# get the current working directory
export CWD=$(pwd)
echo "[$SHELL] ## current working directory: "$CWD

# load modules (assuming you start from the default environment)
# we explicitly call the modules to improve reproducibility
# in case the default settings change
module load Python/3.8.6-GCCcore-10.2.0
echo "module loaded"

source venv/bin/activate
echo "venv activated"

python --version

# Run the file
echo "[$SHELL] ## Run script"
python test.py
echo "test done"

python preprocess.py
echo "[$SHELL] ## Script finished"

echo "[$SHELL] #### Finished Python test. Have a nice day"