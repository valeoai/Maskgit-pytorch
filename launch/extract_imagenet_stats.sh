#!/usr/bin/bash

#SBATCH --job-name=extract_imagenet_stats
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --account EU-25-13
#SBATCH --partition qgpu
#SBATCH --time 12:00:00

cd /home/it4i-vbesnier/Project/Halton-MaskGIT/ || exit
source /mnt/proj1/eu-25-13/victor/venv/bin/activate
export OMP_NUM_THREADS=4

python extract_train_fid.py --data-folder /mnt/proj3/dd-23-130/dataset/ImageNet/ --bsize 128 --num-worker 8