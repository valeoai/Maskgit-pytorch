#!/usr/bin/bash

cd /home/victor/workspace/Halton-MaskGIT || exit

# Path
data_folder="/datasets_local/vbesnier/IN_feat_256_to_24/"
vit_folder="./saved_networks/ImageNet_384_large.pth"
writer_log=""
vqgan_folder="./saved_networks/vq_ds16_c2i.pt"
eval_folder="/datasets_local/ImageNet/"
data="imagenet_feat"

# Model size and compute FLOP
vit_size="large"
img_size=384
f_factor=16
codebook_size=16384
register=1
proj=1
dtype="bfloat16"

# Dataloader
num_workers=8
global_bsize=32
gradacc=1
nb_class=1000

# Learning hyper-parameter
max_iter=1000000
warm_up=2500
lr=1e-4
grad_clip=1
sched_mode="arccos"
dropout=0.1

# sampling
sampler="halton"
sm_temp=1.0
cfg_w=0.5            # (visualization or test-only)
step=32

# halton scheduler only
sched_pow=2
top_k=-1
temp_warmup=1
sm_temp_min=1.

# Confidence Sampler only
r_temp=5

torchrun --standalone --nnodes=1 --nproc_per_node=gpu \
  main.py --data-folder "${data_folder}" --vit-folder "${vit_folder}" --vqgan-folder "${vqgan_folder}" \
  --writer-log "${writer_log}" --num-workers ${num_workers} --mode "cls-to-img" --lr "${lr}" --grad-cum "${gradacc}" \
  --global-bsize "${global_bsize}" --warm-up "${warm_up}" --img-size "${img_size}" --vit-size "${vit_size}" \
  --data "${data}" --dtype "${dtype}" --eval-folder "${eval-folder}" --proj "${proj}"\
  --f-factor "${f_factor}" --codebook-size "${codebook_size}" --mask-value "${codebook_size}" \
  --max-iter "${max_iter}" --sm-temp-min "${sm_temp_min}" --sm-temp "${sm_temp}" --cfg-w "${cfg_w}" \
  --temp-warmup "${temp_warmup}" --step "${step}" --temp-warmup "${temp_warmup}" --top-k "${top_k}" \
  --grad-clip "${grad_clip}" --sched-mode "${sched_mode}" --sampler "${sampler}" --r-temp "${r_temp}" \
  --register "${register}" --dropout "${dropout}" --nb-class "${nb_class}" --sched-pow "${sched_pow}" \
  --eval-folder "${eval_folder}" --resume --compile --test-only