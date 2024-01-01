python3 main_joint.py -c ./configs/joint.yaml -rm train -ng 8 --epochs 30 \
--version "joint" --lr_drop 20 -bs 1 -ws 8 --backbone "video-swin-b" \
-bpp "/mnt/data_16TB/lzy23/pretrained/pretrained_swin_transformer/swin_tiny_patch244_window877_kinetics400_1k.pth"
