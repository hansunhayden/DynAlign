#!/bin/sh

#runai submit  --name rebuttal-uda -i registry.rcp.epfl.ch/imos-hansun/python3.8-cu11.1-torch1.9.1:v9 \
#  --pvc imos-scratch:/home/hansun/nas \
#  --large-shm \
#  -g 1 --command -- bash /home/hansun/nas/hansun/CODE/Diffusion_Repo/SSA/scripts/rebuttal_rcp/uda_cs.sh
##  runai delete job a-maskclip-hrda-mapi-$i


runai submit  --name hrda-new3 -i registry.rcp.epfl.ch/imos-hansun/python3.8-cu11.1-torch1.9.1:v9 \
 --pvc imos-scratch:/home/hansun/nas \
 --large-shm \
 --node-pool default \
 -g 8 --command -- bash /home/hansun/nas/hansun/CODE/Diffusion_Repo/SSA/scripts/seg/HRDA/train.sh

#runai submit  --name v5-mapi-eval -i registry.rcp.epfl.ch/imos-hansun/python3.8-cu11.1-torch1.9.1:v9 \
#--pvc runai-imos-hansun-scratch:/home/hansun/nas \
#--host-ipc \
#-g 1 --command -- bash /home/hansun/nas/hansun/CODE/Diffusion_Repo/SSA/scripts/rcp_scripts/evaluation_mapi.sh