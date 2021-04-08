#!/bin/bash
source /opt/sw/anaconda3/etc/profile.d/conda.sh
conda activate DexiNed
cd /home/matanr/MLography/Segmentation/DexiNed
python run_model.py --model_state test --dataset_dir $1 --base_dir_results $2 --image_width 128 --image_height 128 --trained_model_dir train --test_snapshot 14999