import argparse
import json
from pydoc import visiblename
import random
import time
from pathlib import Path
import pdb
import numpy as np
import torch
from util.jaccard import db_eval_iou
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
import glob
import opts
from tqdm import tqdm

all_model_path = glob.glob('/mnt/data/rvos_model/ensemble/test/*.pth')
all_mask = []
name_list = []
for path in all_model_path:  # , map_location='cpu'
    name_list.append(path.split('/')[-1].split('.')[0])
    res_info = torch.load(path, map_location='cpu')
    all_mask.append(res_info)
    print('load model: {}. Done!'.format(path))

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
device = 'cuda'

output_dir = './models/result'
root = '/mnt/Data/Competition/youtube-VOS-new/2019/image'

test_meta_file = os.path.join(root, "meta_expressions", "test", "meta_expressions.json")
with open(test_meta_file, 'r') as f:
    data = json.load(f)['videos']
test_videos = set(data.keys())

# pdb.set_trace()
video_list = sorted([video for video in test_videos])
# assert len(video_list) == 202, 'error: incorrect number of validation videos'

thr = 0.9
save_path_prefix = os.path.join(output_dir, 'Annotations_' + str(thr).replace('.', '_') + '_6603_649_642')
if not os.path.exists(save_path_prefix):
    os.makedirs(save_path_prefix)
key_frames_id = {}
for video in tqdm(video_list):
    metas = [] # list[dict], length is number of expressions
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
    for i in range(num_expressions):
        video_name = meta[i]["video"]
        # exp = meta[i]["exp"]
        exp_id = meta[i]["exp_id"]
        key_frames_id[video][exp_id] = {}
        frames = meta[i]["frames"]
        video_len = len(frames)
        # tmp_mask = []
        tmp_mask = 0
        # tmp_weight = []
        save_path = os.path.join(save_path_prefix, video_name, exp_id)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
#         
        tmp_mask_0 = all_mask[0][video][exp_id]['mask_logits'].to(device).sigmoid()
        weights_0 = np.array(all_mask[0][video][exp_id]['class_info'])
        
        tmp_mask_1 = all_mask[1][video][exp_id]['mask_logits'].to(device).sigmoid()
        weights_1 = np.array(all_mask[1][video][exp_id]['class_info'])
        
        tmp_mask_2 = all_mask[2][video][exp_id]['mask_logits'].to(device).sigmoid()
        weights_2 = np.array(all_mask[2][video][exp_id]['class_info'])
        
        index = int(np.argmax(weights_0 * weights_1 * weights_2))
        mask_info = {
                'index_info': {'class_index': index}
            }
        key_frames_id[video][exp_id].update(mask_info)
        
        J_i = db_eval_iou(tmp_mask_0[index].squeeze(0).cpu().numpy().astype(np.float32), tmp_mask_1[index].squeeze(0).cpu().numpy().astype(np.float32))
        if J_i > thr:
            tmp_mask_0[index] = (tmp_mask_0[index] + tmp_mask_1[index]) / 2
        
        J_i = db_eval_iou(tmp_mask_0[index].squeeze(0).cpu().numpy().astype(np.float32), tmp_mask_2[index].squeeze(0).cpu().numpy().astype(np.float32))
        if J_i > thr:
            tmp_mask_0[index] = (tmp_mask_0[index] + tmp_mask_2[index]) / 2
        
        pred_masks = (tmp_mask_0 > 0.5).detach().cpu().numpy() 
        # save binary image

        for j in range(video_len):
            frame_name = frames[j]
            mask = pred_masks[j].astype(np.float32) 
            mask = Image.fromarray(mask * 255).convert('L')
            save_file = os.path.join(save_path, frame_name + ".png")
            mask.save(save_file)
            
json_save_path = os.path.join(output_dir, 'key_frame_'+ str(thr).replace('.', '_') + '_6603_649_642'+'.json')
with open(json_save_path, 'w') as f:
    json.dump(key_frames_id, f) 