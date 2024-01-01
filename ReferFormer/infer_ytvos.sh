export CUDA_VISIBLE_DEVICES=0
python3 inference_ytvos.py --with_box_refine --binary --freeze_text_encoder --backbone "video_swin_b_p4w7" \
--resume "/mnt/data_16TB/lzy23/referformer/ytvos_video_swin_base_joint.pth" \
--ytvos_path "/mnt/data_16TB/lzy23/rvosdata/refer_youtube_vos" \
--split "test" \
--ngpu 1