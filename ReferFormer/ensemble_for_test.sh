./scripts/ensemble_test.sh
"/opt/tiger/ljyaronld/referformer/ytvos_video_swin_base_joint.pth" \
--backbone "video_swin_b_p4w7" \
--use_checkpoint \
--cuda_id 0 \
--ensemble_save_path "/opt/tiger/ljyaronld/MTTR/soc_aot/soc/" \
--ytvos_path "/opt/tiger/ljyaronld/rvosdata/refer_youtube_vos" \
--version "referformer_ft.pth"