#!/usr/bin/env bash

EXP_NAME="video_joint_convnext_large"

# OD, IS, REC, RES
python3 launch.py --nn 1 --eval-only \
--uni 1 --config-file projects/UNINEXT/configs/image_joint_convnext_large.yaml \
--resume OUTPUT_DIR outputs/${EXP_NAME}

# VIS
python3 launch.py --nn 1 --eval-only \
--uni 1 --config-file projects/UNINEXT/configs/video_joint_convnext_large.yaml \
--resume OUTPUT_DIR outputs/${EXP_NAME} \
MODEL.USE_IOU_BRANCH False

cd outputs/${EXP_NAME}/inference
zip VIS19.zip results.json
cd ../../..

# OVIS
python3 launch.py --nn 1 --eval-only \
--uni 1 --config-file projects/UNINEXT/configs/eval-vid/video_joint_convnext_large_eval_ovis.yaml \
--resume OUTPUT_DIR outputs/${EXP_NAME} \
MODEL.USE_IOU_BRANCH False

cd outputs/${EXP_NAME}/inference
zip OVIS.zip results.json
cd ../../..

# R-VOS
python3 launch.py --nn 1 --eval-only \
--uni 1 --config-file projects/UNINEXT/configs/eval-vid/video_joint_convnext_large_eval_rvos.yaml \
--resume OUTPUT_DIR outputs/${EXP_NAME}

cd outputs/${EXP_NAME}
zip -r RVOS.zip Annotations
cd ../..

# VOS
python3 launch.py --nn 1 --eval-only \
--uni 1 --config-file projects/UNINEXT/configs/eval-vid/video_joint_convnext_large_eval_vos.yaml \
--resume OUTPUT_DIR outputs/${EXP_NAME} \
SOT.INFERENCE_ON_3F True

cd outputs/${EXP_NAME}/inference
mv ytbvos18 Annotations
zip -r VOS.zip Annotations
cd ../../..

# SOT
python3 launch.py --nn 1 --eval-only \
--uni 1 --config-file projects/UNINEXT/configs/eval-vid/video_joint_convnext_large_eval_sot.yaml \
--resume OUTPUT_DIR outputs/${EXP_NAME} \
DATALOADER.NUM_WORKERS 0 SOT.ONLINE_UPDATE True

# MOT
python3 tools_bin/grid_search_bdd.py --config_name eval-vid/video_joint_convnext_large_eval_mot.yaml --exp_dir_name ${EXP_NAME}
