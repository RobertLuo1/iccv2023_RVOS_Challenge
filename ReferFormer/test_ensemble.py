import argparse
import json
from pathlib import Path
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

import json
import opts
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
    # root = '/mnt/Data/Competition/youtube-VOS-new/2019/image'
    test_meta_file = os.path.join(Path(args.ytvos_path), "meta_expressions", "test", "meta_expressions.json")

    with open(test_meta_file, 'r') as f:
        data = json.load(f)['videos']
    test_videos = set(data.keys())

    video_list = sorted([video for video in test_videos])

    output_dir = args.output_dir

    thr = 4
    split_name = 'AOT_test_thr_4_else_3'

    save_path_prefix = os.path.join(output_dir, 'Annotations_' + split_name)
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    mask_root = args.mask_root

    path_swap = os.path.join('Annotations_swap_6603_649_642', 'Annotations_AOT_class_index')
    path_642 = os.path.join('Annotations_swin_l_642', 'Annotations_AOT_class_index')
    path_649 = os.path.join('test_649', 'Annotations')
    path_6603 = os.path.join('66_test', 'Annotations')


    for video in tqdm(video_list):
        metas = []  # list[dict], length is number of expressions
        expressions = data[video]["expressions"]
        expression_list = list(expressions.keys())
        num_expressions = len(expression_list)
        video_len = len(data[video]["frames"])

        mask_swap_path = os.path.join(mask_root, path_swap, video)
        mask_642_path = os.path.join(mask_root, path_642, video)
        mask_649_path = os.path.join(mask_root, path_649, video)
        mask_6603_path = os.path.join(mask_root, path_6603, video)

        # read all the anno meta
        for i in range(num_expressions):
            meta = {}
            meta["video"] = video
            meta["exp"] = expressions[expression_list[i]]["exp"]
            meta["exp_id"] = expression_list[i]
            meta["frames"] = data[video]["frames"]
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

            mask_swap_path_1 = os.path.join(mask_swap_path, exp_id_1)
            mask_swap_path_2 = os.path.join(mask_swap_path, exp_id_2)

            mask_642_path_1 = os.path.join(mask_642_path, exp_id_1)
            mask_642_path_2 = os.path.join(mask_642_path, exp_id_2)

            mask_649_path_1 = os.path.join(mask_649_path, exp_id_1)
            mask_649_path_2 = os.path.join(mask_649_path, exp_id_2)

            mask_6603_path_1 = os.path.join(mask_6603_path, exp_id_1)
            mask_6603_path_2 = os.path.join(mask_6603_path, exp_id_2)

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
            # swap
            all_swap_masks_1 = read_mask(mask_swap_path_1)
            all_swap_masks_2 = read_mask(mask_swap_path_2)

            # 642
            all_642_masks_1 = read_mask(mask_642_path_1)
            all_642_masks_2 = read_mask(mask_642_path_2)

            # 649
            all_649_masks_1 = read_mask(mask_649_path_1)
            all_649_masks_2 = read_mask(mask_649_path_2)

            # 6603
            all_6603_masks_1 = read_mask(mask_6603_path_1)
            all_6603_masks_2 = read_mask(mask_6603_path_2)

            tmp_mask_swap = all_swap_masks_1 + all_swap_masks_2 + all_642_masks_1 + \
                all_642_masks_2 + all_649_masks_1 + all_649_masks_2 + all_6603_masks_1 + all_6603_masks_2

            pred_masks = np.where(tmp_mask_swap>=thr, 1, 0)
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

            mask_swap_path = os.path.join(mask_root, path_swap, video_name, exp_id)
            all_swap_masks = read_mask(mask_swap_path)

            mask_642_path = os.path.join(mask_root, path_642, video_name, exp_id)
            all_642_masks = read_mask(mask_642_path)

            mask_649_path = os.path.join(mask_root, path_649, video_name, exp_id)
            all_649_masks = read_mask(mask_649_path)

            mask_6603_path = os.path.join(mask_root, path_6603, video_name, exp_id)
            all_6603_masks = read_mask(mask_6603_path)

            tmp_mask_swap = all_swap_masks + all_649_masks + all_6603_masks + all_642_masks
            all_swap_masks = np.where(tmp_mask_swap >= 3, 1, 0)

            for j in range(video_len):
                frame_name = frames[j]
                mask = all_swap_masks[j].astype(np.float32)
                mask = Image.fromarray(mask * 255).convert('L')
                save_file = os.path.join(save_path, frame_name + ".png")
                mask.save(save_file)

    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('ReferFormer inference script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    ensemble(args)