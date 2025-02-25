#!/usr/bin/bash

# bash cmd
cd /home/victor/workspace/github_release/Halton-MaskGIT/ || exit

python extract_vq_features.py \
   --data-folder="/datasets_local/ImageNet/" \
   --dest-folder="/home/victor/scania/vbesnier/logs/IN_feat_256_to_24/" \
   --vqgan-folder="/home/victor/no_backup/logs/VQGAN/vq_ds16_c2i.pt" \
   --bsize=256 --f-factor 16 --compile --img-size 384