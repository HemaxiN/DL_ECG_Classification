#!/bin/bash
#SBATCH --job-name=pipeline_ecg
#SBATCH --output=my-output.log
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --mail-user=ricardo.santos@fraunhofer.pt
#SBATCH --mail-type=ALL

# mkdir -p /hpc/scratch/$user
# load modules
module load ccuda11.6/toolkit/11.6.2

conda activate dl_ecg
. .venv-dev/bin/activate

cd ~/DL_ECG_Classification || return
# run your code (pip install modules on login node; datasets read directly from /net/sharedfolders/datasets)
# dvc repro
python multimodal_training.py -m

# copy results back to your home
# cp -r /hpc/scratch/$user/my-results ~/my-results

# delete scratch
# rm -rf /hpc/scratch/$user
