import os
import json
import torch
from tqdm import tqdm
import numpy as np

ytvos_path = "/mnt/data_16TB/lzy23/rvosdata/refer_youtube_vos"
model_name = "mutr"
model_dir = "/mnt/data_16TB/lzy23/test/model_pth"
output_dir = "/mnt/data_16TB/lzy23/test/mutr"
model_pth = os.path.join(model_dir, model_name + ".pth")
# valid_meta_file = os.path.join(ytvos_path, "meta_expressions", "valid", "meta_expressions.json")
# with open(valid_meta_file, "r") as f:
#     data = json.load(f)["videos"]
# valid_test_videos = set(data.keys())
test_meta_file = os.path.join(ytvos_path, "meta_expressions", "test", "meta_expressions.json")
with open(test_meta_file, 'r') as f:
    test_data = json.load(f)['videos']
test_videos = set(test_data.keys())
# valid_videos = valid_test_videos - test_videos
video_list = sorted([video for video in test_videos])
# video_list = sorted([video for video in valid_videos])
masks_info = torch.load(model_pth, map_location="cpu")

key_frames_id = {}
count = 0
for idx, video in enumerate(tqdm(video_list)):
    metas = []
    key_frames_id[video] = {}
    expressions = test_data[video]["expressions"]   
    expression_list = list(expressions.keys()) 
    num_expressions = len(expression_list)

    for i in range(num_expressions):
        meta = {}
        meta["video"] = video
        meta["exp"] = expressions[expression_list[i]]["exp"]
        meta["exp_id"] = expression_list[i]
        meta["frames"] = test_data[video]["frames"]
        metas.append(meta)
    meta = metas
    
    for i in range(num_expressions):
        video_name = meta[i]["video"]
        exp = meta[i]["exp"]
        exp_id = meta[i]["exp_id"]

        # # if video_name == '6a75316e99':
        # if video_name == "0062f687f1":
        #     print("ok")
        #     count += 1
        key_frames_id[video][exp_id] = {}
        # import pdb; pdb.set_trace()
        weight = np.array(masks_info[video_name][exp_id]['class_info'])
        index = int(np.argmax(weight))
        mask_info = {
                'index_info': {'class_index': index}
            }
        key_frames_id[video][exp_id].update(mask_info)
# print(count)
json_save_path = os.path.join(output_dir, 'key_frame' + '.json')
with open(json_save_path, 'w') as f:
    json.dump(key_frames_id, f)
print('Done!')