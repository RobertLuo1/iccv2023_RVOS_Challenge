export CUDA_VISIBLE_DEVICES=4,5
python3 main.py -c ./configs/refer_youtube_vos.yaml -rm train -ng 2 --epochs 30 \
--version "ref_68" --lr_drop 20 -bs 1 -ws 8 --backbone "video-swin-t" \
-bpp "/mnt/data_16TB/lzy23/pretrained/pretrained_swin_transformer/swin_tiny_patch244_window877_kinetics400_1k.pth"

# python main.py -c ./configs/refer_youtube_vos.yaml -rm train -ng 8 --epochs 15 \
# --version "ref_69" --lr_drop 8 12
