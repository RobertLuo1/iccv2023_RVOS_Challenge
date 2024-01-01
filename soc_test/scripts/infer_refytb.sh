export CUDA_VISIBLE_DEVICES=0
python infer_refytb.py -c ./configs/refer_youtube_vos.yaml -rm test --version "soc_test" -ng 1 --backbone "video-swin-b" \
-bpp "/mnt/data_16TB/lzy23/pretrained/pretrained_swin_transformer/swin_base_patch244_window877_kinetics400_22k.pth" \
-ckpt "/mnt/data_16TB/lzy23/MTTR/soc/joint_base.tar"

### 3 0.8 0.2 4 0.7 0.3 5 0.2 0.8 6 0.3 0.7(video frame)