# #!/usr/bin/env bash

# inference
python3 inference_ytvos_for_ensemble.py --with_box_refine --binary --freeze_text_encoder \
--resume "/mnt/data_16TB/lzy23/referformer/ytvos_video_swin_base_joint.pth" \
--backbone "video_swin_b_p4w7" \
--use_checkpoint \
--cuda_id 0 \
--ensemble_save_path "/mnt/data_16TB/lzy23/test/model_pth" \
--ytvos_path "/mnt/data_16TB/lzy23/rvosdata/refer_youtube_vos" \
--version "referformer_ft.pth"