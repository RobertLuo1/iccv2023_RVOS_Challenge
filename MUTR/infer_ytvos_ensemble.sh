python3 infer_ytvos_ensemble.py --with_box_refine --binary --freeze_text_encoder \
--backbone "video_swin_b_p4w7" --cuda_id 0 --version "mutr.pth" \
--split "test" \
--resume "/mnt/data_16TB/lzy23/MUTR/video_swin_b_p4w7.pth" \
--ytvos_path "/mnt/data_16TB/lzy23/rvosdata/refer_youtube_vos" \
--ensemble_save_path "/mnt/data_16TB/lzy23/test/model_pth"