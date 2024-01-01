'''
Inference code for ReferFormer, on Ref-Youtube-VOS
Modified from DETR (https://github.com/facebookresearch/detr)
'''
import argparse
import json
import random
import time
from pathlib import Path
import pdb
import numpy as np
import torch
import util.misc as utils
from models import build_model
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image, ImageDraw, ImageFont
import math
import torch.nn.functional as F
import json
import opts
from tqdm import tqdm
import multiprocessing as mp
import threading

from tools.colormap import colormap

# colormap
color_list = colormap()
color_list = color_list.astype('uint8').tolist()

# build transform
transform = T.Compose([
    T.Resize(360),
    # T.
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def main(args):
    args.masks = True
    args.batch_size == 1
    print("Inference only supports for batch size = 1") 

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # load data
    root = Path(args.ytvos_path)  # data/ref-youtube-vos
    img_folder = os.path.join(root, "JPEGImages")
    test_meta_file = os.path.join(root, "meta_expressions", "test", "meta_expressions.json")
    with open(test_meta_file, 'r') as f:
        test_data = json.load(f)['videos']
    test_videos = set(test_data.keys())

    valid_videos = test_videos
    video_list = sorted([video for video in valid_videos])
    start_time = time.time()
    print('Start inference')

    with torch.cuda.amp.autocast(enabled=True):
        sub_processor(args, test_data, img_folder, video_list)

    end_time = time.time()
    total_time = end_time - start_time
    print("Total inference time: %.4f s" %(total_time))

def sub_processor(args, data, img_folder, video_list):

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_id
    # model
    model, criterion, _ = build_model(args) 
    device = args.device
    model.to(device)

    model_without_ddp = model

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
    else:
        raise ValueError('Please specify the checkpoint for inference.')

    save_path_prefix = os.path.join(args.output_dir, 'Annotations_' + args.part_name)
    if not os.path.exists(save_path_prefix):
        os.makedirs(save_path_prefix)

    # start inference
    model.eval()
    key_frames_id = {}

    # 1. For each video
    for video in tqdm(video_list):
        metas = []  # list[dict], length is number of expressions
        key_frames_id[video] = {}
        expressions = data[video]["expressions"]   
        expression_list = list(expressions.keys()) 
        num_expressions = len(expression_list)

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
        for i in range(num_expressions):
            video_name = meta[i]["video"]
            exp = meta[i]["exp"]
            exp_id = meta[i]["exp_id"]
            frames = meta[i]["frames"]
            key_frames_id[video][exp_id] = {}
            video_len = len(frames)
            # store images
            imgs = []
            for t in range(video_len):
                frame = frames[t]
                img_path = os.path.join(img_folder, video_name, frame + ".jpg")
                img = Image.open(img_path).convert('RGB')
                origin_w, origin_h = img.size
                imgs.append(transform(img))  # list[img]
            imgs = torch.stack(imgs, dim=0).to(args.device)  # [video_len, 3, h, w]
            img_h, img_w = imgs.shape[-2:]
            size = torch.as_tensor([int(img_h), int(img_w)]).to(args.device)
            target = {"size": size}
            with torch.cuda.amp.autocast(enabled=True):
                with torch.no_grad():
                    outputs = model([imgs], [exp], [target])
            pred_logits = outputs["pred_logits"][0]
            pred_masks = outputs["pred_masks"][0]

            # according to pred_logits, select the query index
            tmp = pred_logits.sigmoid()  # [t, q, k]
            pred_scores = tmp.mean(0)
            max_scores, _ = pred_scores.max(-1)  # [q,]
            _, max_ind = max_scores.max(-1)     # [1,]
            max_inds = max_ind.repeat(video_len)

            class_info = tmp[range(video_len), max_inds, ...].tolist()
            class_info = [cl[0] for cl in class_info]
            pred_masks = pred_masks[range(video_len), max_inds, ...]  # [t, h, w]
            pred_masks = pred_masks.unsqueeze(0)
            pred_masks = F.interpolate(pred_masks, size=(origin_h, origin_w), mode='bilinear', align_corners=False)
            pred_masks = pred_masks.sigmoid()

            class_index = pick_key_frames_id(class_info)
            print('Prcessing Seq {}-{} picked frames id from class score: {}.png'.format(video_name, exp_id, frames[class_index]))
            mask_info = {
                'index_info': {'class_index': class_index}
            }

            key_frames_id[video][exp_id].update(mask_info)
            pred_masks = (pred_masks > args.threshold).squeeze(0).detach().cpu().numpy()
            all_pred_masks = pred_masks
            # save binary image
            save_path = os.path.join(save_path_prefix, video_name, exp_id)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            for j in range(video_len):
                frame_name = frames[j]
                mask = all_pred_masks[j].astype(np.float32) 
                mask = Image.fromarray(mask * 255).convert('L')
                save_file = os.path.join(save_path, frame_name + ".png")
                mask.save(save_file)

    json_save_path = os.path.join(args.output_dir, 'key_frame_' + args.part_name + '.json')
    with open(json_save_path, 'w') as f:
        json.dump(key_frames_id, f)

def pick_key_frames_id(class_info):
    class_info = np.array(class_info)
    class_index = np.argmax(class_info)
    return int(class_index)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ReferFormer inference script', parents=[opts.get_args_parser()])
    args = parser.parse_args()
    main(args)
