./scripts/dist_train_test_ytvos.sh 
[/path/to/output_dir] \
[/path/to/pretrained_weight] \
--backbone video_swin_b_p4w7 \
--use_checkpoint \
--dataset_file ytvos \
--lr 5e-6 \
--epochs 6 --lr_drop 3 5 \
--lr_backbone 1e-6 \
--lr_text_encoder 1e-6 \
