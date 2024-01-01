import argparse
import json
from pydoc import visiblename
import random
import time
from pathlib import Path
import numpy as np
import torch
import os

from PIL import Image, ImageDraw, ImageFont
import json
import glob
from tqdm import tqdm


def swap(args):
    model_names = ["soc", "mutr", "referformer_ft"]
    # all_model_path = [os.path.join(args.pth_model_path, name + ".pth") for name in model_names]
    # all_model_path = glob.glob(os.path.join(args.pth_model_path, '*.pth'))
    all_mask = []
    # name_list = []
    # for path in all_model_path:  # , map_location='cpu'
    #     name_list.append(path.split('/')[-1].split('.')[0])
    #     res_info = torch.load(path, map_location='cpu')
    #     all_mask.append(res_info)
    #     print('load model: {}. Done!'.format(path))
    device = 'cuda'
    
    ### load data
    # valid_meta_file = os.path.join(Path(args.ytvos_path), "meta_expressions", "valid", "meta_expressions.json")
    # with open(valid_meta_file, "r") as f:
    #     data = json.load(f)["videos"]
    # valid_test_videos = set(data.keys())
    test_meta_file = os.path.join(Path(args.ytvos_path), "meta_expressions", "test", "meta_expressions.json")
    with open(test_meta_file, 'r') as f:
        data = json.load(f)['videos']
    test_videos = set(data.keys())
    # valid_videos = valid_test_videos - test_videos
    video_list = sorted([video for video in test_videos])
    # video_list = sorted([video for video in valid_videos])
    length = len(video_list)
    output_dir = args.output_dir
    split = 5
    change = length // split
    count = 0
    # thr = 4
    thr = 3
    # thr = 2
    # split_name = 'swap_soc_mutr_referft_3'
    save_path_prefix = os.path.join(output_dir, 'Annotations')
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)
    model_paths = [os.path.join(args.pth_model_path, name, name + "{}.pth".format(count)) for name in model_names]
    for model_path in model_paths:
        all_mask.append(torch.load(model_path, map_location="cpu"))
        print("load_model_from {}".format(model_path))
    count +=1

    key_frames_id = {}
    for idx, video in enumerate(tqdm(video_list)):
        metas = []  # list[dict], length is number of expressions
        key_frames_id[video] = {}
        expressions = data[video]["expressions"]
        expression_list = list(expressions.keys())
        num_expressions = len(expression_list)
        video_len = len(data[video]["frames"])
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

            key_frames_id[video][exp_id_1] = {}
            key_frames_id[video][exp_id_2] = {}

            frames_1 = meta[idx_1]["frames"]
            frames_2 = meta[idx_2]["frames"]

            video_len_1 = len(frames_1)
            video_len_2 = len(frames_2)

            save_path_1 = os.path.join(save_path_prefix, video_name_1, exp_id_1)
            save_path_2 = os.path.join(save_path_prefix, video_name_2, exp_id_2)

            if not os.path.exists(save_path_1):
                os.makedirs(save_path_1)
            if not os.path.exists(save_path_2):
                os.makedirs(save_path_2)

            if model_names[0] == "vit_huge" or model_names[0] == "convnext": ###vit has done sigmoid before
                tmp_mask_0_1 = (all_mask[0][video_name_1][exp_id_1]['mask_logits'].to(
                device) > 0.5).detach().cpu().numpy().astype(np.float32)
                tmp_mask_0_2 = (all_mask[0][video_name_2][exp_id_2]['mask_logits'].to(
                device) > 0.5).detach().cpu().numpy().astype(np.float32)
            else:
            # make sure all_mask[0] is you best model, key frame is based on this
                tmp_mask_0_1 = (all_mask[0][video_name_1][exp_id_1]['mask_logits'].to(
                    device).sigmoid() > 0.5).detach().cpu().numpy().astype(np.float32)
                tmp_mask_0_2 = (all_mask[0][video_name_2][exp_id_2]['mask_logits'].to(
                device).sigmoid() > 0.5).detach().cpu().numpy().astype(np.float32)
            
            weights_0_1 = np.array(all_mask[0][video_name_1][exp_id_1]['class_info'])
            weights_0_2 = np.array(all_mask[0][video_name_2][exp_id_2]['class_info'])

            if model_names[1] == "vit_huge" or model_names[1] == "convnext":
                tmp_mask_1_1 = (all_mask[1][video_name_1][exp_id_1]['mask_logits'].to(
                    device) > 0.5).detach().cpu().numpy().astype(np.float32)

                tmp_mask_1_2 = (all_mask[1][video_name_2][exp_id_2]['mask_logits'].to(
                    device) > 0.5).detach().cpu().numpy().astype(np.float32)
            else:
                tmp_mask_1_1 = (all_mask[1][video_name_1][exp_id_1]['mask_logits'].to(
                    device).sigmoid() > 0.5).detach().cpu().numpy().astype(np.float32)
                tmp_mask_1_2 = (all_mask[1][video_name_2][exp_id_2]['mask_logits'].to(
                    device).sigmoid() > 0.5).detach().cpu().numpy().astype(np.float32)
            if model_names[2] == "vit_huge" or model_names[2] == "convnext":
                tmp_mask_2_1 = (all_mask[2][video_name_1][exp_id_1]['mask_logits'].to(
                    device) > 0.5).detach().cpu().numpy().astype(np.float32)
                tmp_mask_2_2 = (all_mask[2][video_name_2][exp_id_2]['mask_logits'].to(
                    device) > 0.5).detach().cpu().numpy().astype(np.float32)
            else:
                tmp_mask_2_1 = (all_mask[2][video_name_1][exp_id_1]['mask_logits'].to(
                    device).sigmoid() > 0.5).detach().cpu().numpy().astype(np.float32)
                tmp_mask_2_2 = (all_mask[2][video_name_2][exp_id_2]['mask_logits'].to(
                    device).sigmoid() > 0.5).detach().cpu().numpy().astype(np.float32)

            # if model_names[3] == "vit_huge" or model_names[3] == "convnext":
            #     tmp_mask_3_1 = (all_mask[3][video_name_1][exp_id_1]['mask_logits'].to(
            #         device) > 0.5).detach().cpu().numpy().astype(np.float32)
            #     tmp_mask_3_2 = (all_mask[3][video_name_2][exp_id_2]['mask_logits'].to(
            #         device) > 0.5).detach().cpu().numpy().astype(np.float32)

            # else:
            #     tmp_mask_3_1 = (all_mask[3][video_name_1][exp_id_1]['mask_logits'].to(
            #         device).sigmoid() > 0.5).detach().cpu().numpy().astype(np.float32)
            #     tmp_mask_3_2 = (all_mask[3][video_name_2][exp_id_2]['mask_logits'].to(
            #         device).sigmoid() > 0.5).detach().cpu().numpy().astype(np.float32)
            
            # if model_names[4] == "vit_huge" or model_names[4] == "convnext":
            #     tmp_mask_4_1 = (all_mask[4][video_name_1][exp_id_1]['mask_logits'].to(
            #         device) > 0.5).detach().cpu().numpy().astype(np.float32)
            #     tmp_mask_4_2 = (all_mask[4][video_name_2][exp_id_2]['mask_logits'].to(
            #         device) > 0.5).detach().cpu().numpy().astype(np.float32)

            # else:
            #     tmp_mask_4_1 = (all_mask[4][video_name_1][exp_id_1]['mask_logits'].to(
            #         device).sigmoid() > 0.5).detach().cpu().numpy().astype(np.float32)
            #     tmp_mask_4_2 = (all_mask[4][video_name_2][exp_id_2]['mask_logits'].to(
            #         device).sigmoid() > 0.5).detach().cpu().numpy().astype(np.float32)

            index_1 = int(np.argmax(weights_0_1))
            mask_info = {
                'index_info': {'class_index': index_1}
            }
            key_frames_id[video][exp_id_1].update(mask_info)

            index_2 = int(np.argmax(weights_0_2))
            mask_info = {
                'index_info': {'class_index': index_2}
            }
            key_frames_id[video][exp_id_2].update(mask_info)
            tmp_mask_6603 = tmp_mask_0_1 + tmp_mask_0_2 + tmp_mask_1_1 + tmp_mask_1_2 + tmp_mask_2_1 + tmp_mask_2_2 \
            #    + tmp_mask_3_1 + tmp_mask_3_2 + tmp_mask_4_1 + tmp_mask_4_2

            pred_masks = np.where(tmp_mask_6603 >= thr, 1, 0)

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

            video_name = meta[num_expressions - 1]["video"]
            exp_id = meta[num_expressions - 1]["exp_id"]
            key_frames_id[video_name][exp_id] = {}

            frames = meta[num_expressions - 1]["frames"]

            save_path = os.path.join(save_path_prefix, video_name, exp_id)

            if not os.path.exists(save_path):
                os.makedirs(save_path)
            tmp_mask = (
                        all_mask[0][video][exp_id]['mask_logits'].to(device).sigmoid() > 0.5).detach().cpu().numpy().astype(
                np.float32)
            weights = np.array(all_mask[0][video][exp_id]['class_info'])

            index = int(np.argmax(weights))
            mask_info = {
                'index_info': {'class_index': index}
            }
            key_frames_id[video][exp_id].update(mask_info)
            for j in range(video_len):
                frame_name = frames[j]
                mask = tmp_mask[j]
                mask = Image.fromarray(mask * 255).convert('L')
                save_file = os.path.join(save_path, frame_name + ".png")
                mask.save(save_file)
        if (idx + 1) % change == 0 and (idx + 1) // change != split:
            all_mask.clear()
            model_paths = [os.path.join(args.pth_model_path, name, name + "{}.pth".format(count)) for name in model_names]
            for id, model_path in enumerate(model_paths):
                all_mask.append(torch.load(model_path, map_location="cpu"))
                print("load_model_from {}".format(model_path))
            count +=1

    json_save_path = os.path.join(output_dir, 'key_frame.json')
    with open(json_save_path, 'w') as f:
        json.dump(key_frames_id, f)
    print('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Ensemble test script')
    parser.add_argument("--ytvos_path", type=str)
    parser.add_argument("--pth_model_path", type=str)
    parser.add_argument("--output_dir", type=str)
    args = parser.parse_args()
    swap(args)