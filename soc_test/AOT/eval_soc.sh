exp="soc"
gpu_num="2"

# model="aott"
# model="aots"
# model="aotb"
# model="aotl"
# model="r50_aotl"
# model="swinb_aotl"
model="swinl_aotl"

stage="pre_ytb_dav"


dataset="youtubevos2019_post"
split="valid"  # or "val_all_frames"
python tools/eval.py --exp_name ${exp} --stage ${stage} --model ${model} \
	--dataset ${dataset} --split ${split} --gpu_num ${gpu_num} --flip \
	--ms 0.8 1.0 1.2 --amp --gpu_id 0 \

	# --amp --gpu_id 2
	# --ms 0.8 1.0 1.2 --amp