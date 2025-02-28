#!/bin/bash --login

pwd
#source /home/hansun/.bashrc
conda env list
ls .conda/envs/
source activate base
conda activate ssa
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install transformers==4.29.2
pip install safetensors==0.3.0
pip install ftfy
#pip install mmcv-full==1.3.7 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
cd /home/hansun/nas/hansun/CODE/Diffusion_Repo/SSA/scripts/seg/HRDA
#python -u evaluation.py \
#--gt_path /home/hansun/nas/hansun/DATA/segmentation/cityscapes/gtFine/val \
#--result_path /home/hansun/nas/hansun/OUTPUT/SSA/rebuttal_mask2former_mapi2cs/labels \
#--dataset cityscapes
##--class_set "unknown"

python run_experiments.py --config configs/hrda/mapi2cs.py
#\
#--resume-from /home/hansun/nas/hansun/CODE/Diffusion_Repo/SSA/scripts/seg/HRDA/work_dirs/local-basic/241122_0006_gtaHR2csHR_hrda_s1_a31c3/latest.pth