import argparse
import json
from pathlib import Path
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

import json
import glob
from tqdm import tqdm

def read_mask(mask_path):
    all_path = glob.glob(mask_path + '/*.png')
    all_path.sort()
    all_masks = []
    for tmp_path in all_path:
        mask = Image.open(tmp_path).convert('P')
        mask = np.array(mask)
        mask = (mask == 255).astype(np.float32)
        all_masks.append(mask)
    all_masks = np.array(all_masks)
    return all_masks

def ensemble(args):
    # valid_meta_file = os.path.join(Path(args.ytvos_path), "meta_expressions", "valid", "meta_expressions.json")
    # with open(valid_meta_file, "r") as f:
    #     data = json.load(f)["videos"]
    # valid_videos = set(data.keys())
    # root = '/mnt/Data/Competition/youtube-VOS-new/2019/image'
    test_meta_file = os.path.join(Path(args.ytvos_path), "meta_expressions", "test", "meta_expressions.json")

    with open(test_meta_file, 'r') as f:
        test_data = json.load(f)['videos']
    test_videos = set(test_data.keys())

    # valid_videos = valid_videos - test_videos
    video_list = sorted([video for video in test_videos])
    # video_list  =sorted([video for video in valid_videos])

    output_dir = args.output_dir

    thr = 4
    split_name = args.split_name

    save_path_prefix = os.path.join(output_dir, 'Annotations_' + split_name)
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    mask_root = args.mask_root

    path_0 = os.path.join("soc_mutr_referft", "Annotations_AOT_class_index")
    path_1 = os.path.join("soc", "Annotations_AOT_class_index")
    path_2 = os.path.join("mutr", "Annotations_AOT_class_index")
    path_3 = os.path.join("referformer_ft", "Annotations_AOT_class_index")


    for video in tqdm(video_list):
        metas = []  # list[dict], length is number of expressions
        expressions = test_data[video]["expressions"]
        expression_list = list(expressions.keys())
        num_expressions = len(expression_list)
        video_len = len(test_data[video]["frames"])

        mask_0_path = os.path.join(mask_root, path_0, video)
        mask_1_path = os.path.join(mask_root, path_1, video)
        # mask_soc_path = os.path.join(mask_root, path_soc, video)
        # mask_mutr_path = os.path.join(mask_root, path_mutr, video)
        mask_2_path = os.path.join(mask_root, path_2, video)
        mask_3_path = os.path.join(mask_root, path_3, video)

        # read all the anno meta
        for i in range(num_expressions):
            meta = {}
            meta["video"] = video
            meta["exp"] = expressions[expression_list[i]]["exp"]
            meta["exp_id"] = expression_list[i]
            meta["frames"] = test_data[video]["frames"]
            metas.append(meta)
        meta = metas

        # 2. For each expression
        lan_len = num_expressions // 2
        remain = num_expressions % 2
        for i in range(lan_len):
            idx_1 = i * 2
            idx_2 = idx_1 + 1

            video_name_1 = meta[idx_1]["video"]
            video_name_2 = meta[idx_2]["video"]

            # exp = meta[i]["exp"]
            exp_id_1 = meta[idx_1]["exp_id"]
            exp_id_2 = meta[idx_2]["exp_id"]

            mask_0_path_1 = os.path.join(mask_0_path, exp_id_1)
            mask_0_path_2 = os.path.join(mask_0_path, exp_id_2)

            mask_1_path_1 = os.path.join(mask_1_path, exp_id_1)
            mask_1_path_2 = os.path.join(mask_1_path, exp_id_2)

            mask_2_path_1 = os.path.join(mask_2_path, exp_id_1)
            mask_2_path_2 = os.path.join(mask_2_path, exp_id_2)

            # mask_soc_path_1 = os.path.join(mask_soc_path, exp_id_1)
            # mask_soc_path_2 = os.path.join(mask_soc_path, exp_id_2)

            # mask_mutr_path_1 = os.path.join(mask_mutr_path, exp_id_1)
            # mask_mutr_path_2 = os.path.join(mask_mutr_path, exp_id_2)

            mask_3_path_1 = os.path.join(mask_3_path, exp_id_1)
            mask_3_path_2 = os.path.join(mask_3_path, exp_id_2)

            frames_1 = meta[idx_1]["frames"]
            frames_2 = meta[idx_2]["frames"]

            video_len_1 = len(frames_1)
            video_len_2 = len(frames_2)

            # tmp_mask = []
            tmp_mask = 0
            # tmp_weight = []
            save_path_1 = os.path.join(save_path_prefix, video_name_1, exp_id_1)
            save_path_2 = os.path.join(save_path_prefix, video_name_2, exp_id_2)

            if not os.path.exists(save_path_1):
                os.makedirs(save_path_1)
            if not os.path.exists(save_path_2):
                os.makedirs(save_path_2)
            
            all_0_masks_1 = read_mask(mask_0_path_1)
            all_0_masks_2 = read_mask(mask_0_path_2)

            # swap
            all_1_masks_1 = read_mask(mask_1_path_1)
            all_1_masks_2 = read_mask(mask_1_path_2)

            # 66.1
            all_2_masks_1 = read_mask(mask_2_path_1)
            all_2_masks_2 = read_mask(mask_2_path_2)

            # 674
            # all_soc_masks_1 = read_mask(mask_soc_path_1)
            # all_soc_masks_2 = read_mask(mask_soc_path_2)

            # 675
            # all_mutr_masks_1 = read_mask(mask_mutr_path_1)
            # all_mutr_masks_2 = read_mask(mask_mutr_path_2)
            all_3_masks_1 = read_mask(mask_3_path_1)
            all_3_masks_2 = read_mask(mask_3_path_2)

            # import pdb; pdb.set_trace()
            tmp_mask_swap = all_0_masks_1 + all_0_masks_2 +  all_1_masks_1 + all_1_masks_2 \
             + all_2_masks_1 + all_2_masks_2 +  all_3_masks_1 + all_3_masks_2
             

            pred_masks = np.where(tmp_mask_swap >= thr, 1, 0)
            # save binary image
            for j in range(video_len_1):
                frame_name = frames_1[j]
                mask = pred_masks[j].astype(np.float32)
                mask = Image.fromarray(mask * 255).convert('L')
                save_file = os.path.join(save_path_1, frame_name + ".png")
                mask.save(save_file)

            for j in range(video_len_2):
                frame_name = frames_2[j]
                mask = pred_masks[j].astype(np.float32)
                mask = Image.fromarray(mask * 255).convert('L')
                save_file = os.path.join(save_path_2, frame_name + ".png")
                mask.save(save_file)

        if remain != 0:

            video_name = meta[num_expressions-1]["video"]

            exp_id = meta[num_expressions-1]["exp_id"]
    #         key_frames_id[video_name][exp_id] = {}

            frames = meta[num_expressions-1]["frames"]

            save_path = os.path.join(save_path_prefix, video_name, exp_id)

            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            mask_0_path = os.path.join(mask_root, path_0, video_name, exp_id)
            all_0_masks = read_mask(mask_0_path)

            mask_1_path = os.path.join(mask_root, path_1, video_name, exp_id)
            all_1_masks = read_mask(mask_1_path)

            mask_2_path = os.path.join(mask_root, path_2, video_name, exp_id)
            all_2_masks = read_mask(mask_2_path)

            # mask_soc_path = os.path.join(mask_root, path_soc, video_name, exp_id)
            # all_soc_masks = read_mask(mask_soc_path)

            # mask_mutr_path = os.path.join(mask_root, path_mutr, video_name, exp_id)
            # all_mutr_masks = read_mask(mask_mutr_path)
            mask_3_path = os.path.join(mask_root, path_3, video_name, exp_id)
            all_3_masks = read_mask(mask_3_path)

            # tmp_mask_swap = all_swap_masks + all_soc_masks + all_mutr_masks + all_refft_masks
            # tmp_mask_swap = all_vith_masks + all_swap_masks + all_mutr_masks + all_refft_masks
            tmp_mask_swap = all_0_masks + all_1_masks + all_2_masks + all_3_masks
            all_swap_masks = np.where(tmp_mask_swap >= 3, 1, 0)

            for j in range(video_len):
                frame_name = frames[j]
                mask = all_swap_masks[j].astype(np.float32)
                mask = Image.fromarray(mask * 255).convert('L')
                save_file = os.path.join(save_path, frame_name + ".png")
                mask.save(save_file)

    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Ensemble test script')
    parser.add_argument("--ytvos_path", type=str)
    parser.add_argument("--mask_root", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--split_name", type=str)
    args = parser.parse_args()
    args = parser.parse_args()
    ensemble(args)