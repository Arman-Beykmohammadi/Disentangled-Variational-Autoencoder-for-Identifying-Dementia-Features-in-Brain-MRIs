#!/bin/bash -eux

#SBATCH --job-name=IVAE
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=Arman.Beykmohammadi@slack
#SBATCH --partition=gpupro,gpua100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40gb
#SBATCH --time=3-00:00:00
#SBATCH -o logs/IVAE/%x_%A_%a.log

# Your Conda environment name
ENV_NAME=pytorch

# Initialize Conda environment
source /dhc/home/arman.beykmohammadi/conda3/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Now run the Python script
python src/main.py

# Optionally deactivate the environment
conda deactivate
