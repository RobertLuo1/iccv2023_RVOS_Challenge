import os
import torch
from tqdm import tqdm

pth_model_dir = "/mnt/data_16TB/lzy23/MTTR/test/model_pth"
output_split_dir = "/mnt/data_16TB/lzy23/MTTR/test/model_split_2"
# model_names = ["soc", "mutr", "referformer_ft"]
model_names = ["soc", "mutr", "referformer_ft", "vit_huge", "convnext"]
# model_names = ["vit_huge"]
model_paths = [os.path.join(pth_model_dir, name + ".pth") for name in model_names]
split_num = 5
for i, model_path in tqdm(enumerate(model_paths)):
    save_dir = os.path.join(output_split_dir, model_names[i])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    count = 0
    model_info = torch.load(model_path, map_location="cpu")
    print('load model: {}. Done!'.format(model_path))
    length = len(model_info.keys())
    parts_num = length // split_num
    print(parts_num)
    remains = length % split_num
    temp_dict = {}
    for idx, (video_ids, model_info) in enumerate(model_info.items()):
        if (idx + 1) % parts_num == 0:
            temp_dict[video_ids] = model_info
            save_path = os.path.join(save_dir, model_names[i] + "{}.pth".format(count))
            torch.save(temp_dict, save_path)
            count +=1
            temp_dict.clear()
        else:
            temp_dict[video_ids] = model_info
    if remains != 0:
        save_path = os.path.join(save_dir, model_names[i] + "{}.pth".format(count))
        torch.save(temp_dict, save_path)
print("DONE")

