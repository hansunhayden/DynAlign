#!/bin/bash --login

python -u dynalign_mapillary.py \
--eval mIoU \
--eval-option efficient_test=True \
--world_size 1 \
--dataset Mapillary \
--dataset_sam mapillary_openset \
--data_dir ./data/mapillary_vistas/validation \
--sam_path ./checkpoints/sam_vit_h_4b8939.pth \
--config ./checkpoints/seg_hrda/seg_hrda.json \
--checkpoint ./checkpoints/seg_hrda/iter_40000_relevant.pth \
--save_img \
--save_original_seg \
--save_gt \
--out_dir ./mapi_result \
--prob_threshold 0.6
