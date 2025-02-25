#!/usr/bin/bash

#SBATCH --job-name=train_noise_big
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --account DD-23-55
#SBATCH --partition qgpu
#SBATCH --time 48:00:00

cd /home/it4i-vbesnier/Project/MaskGIT-pytorch/ || exit

source activate maskgit
module load CUDA/11.7.0
module load cuDNN/8.4.1.50-CUDA-11.7.0
echo $CONDA_PREFIX

# Your script goes here
data_folder="/mnt/proj2/dd-23-55/dataset/ImageNet/"
vit_folder="/mnt/proj2/dd-23-55/victor/MaskGIT/saved_networks/train_noise_big_more/"
vqgan_folder="/mnt/proj2/dd-23-55/victor/VQGAN/saved_networks/vqgan_imagenet_f16_1024/"
writter_log="/mnt/proj2/dd-23-55/victor/MaskGIT/train_noise_big_more/"
num_worker=16
torchrun --standalone --nnodes=1 --nproc_per_node=gpu main.py --bsize 48 --data-folder "${data_folder}" --vit-folder "${vit_folder}" --vqgan-folder "${vqgan_folder}" --writer-log "${writter_log}" --num_workers ${num_worker} --lr 1e-4 --img-size 256 --epoch 600 --vit-size "big" --mask-value -1 --grad-cum 2 --resume --sm_temp 1.15 --r_temp 7 --cfg_w 9 --step 32
