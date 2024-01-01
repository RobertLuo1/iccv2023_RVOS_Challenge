export CUDA_VISIBLE_DEVICES=1
python3 launch.py --nn 1 --eval-only --np 1 \
--uni 1 --config-file projects/UNINEXT/configs/eval-vid/video_joint_r50_eval_rvos.yaml \
--resume 