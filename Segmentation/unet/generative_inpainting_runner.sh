#!/bin/bash
source /opt/sw/anaconda3/etc/profile.d/conda.sh
conda activate generative_inpainting
cd /home/matanr/MLography/Segmentation/generative_inpainting
python test.py --image $1 --mask $2 --output $3 --checkpoint_dir model_logs