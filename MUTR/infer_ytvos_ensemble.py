'''
Inference code for ReferFormer, on Ref-Youtube-VOS
Modified from DETR (https://github.com/facebookresearch/detr)
'''
import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import util.misc as utils
from models import build_model
import torchvision.transforms as T
import os
from PIL import Image
import torch.nn.functional as F
import json
import opts
from tqdm import tqdm
from tools.colormap import colormap


# colormap
color_list = colormap()
color_list = color_list.astype('uint8').tolist()

# build transform
transform = T.Compose([
    T.Resize(360),
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
    split = args.split

    # load data,
    root = Path(args.ytvos_path)  # data/ref-youtube-vos
    img_folder = os.path.join(root, split, "JPEGImages")

    # part code of valid

    # meta_file = os.path.join(root, "meta_expressions", split, "meta_expressions.json")
    # with open(meta_file, "r") as f:
    #     data = json.load(f)["videos"]
    # valid_test_videos = set(data.keys())

    # for some reasons the competition's validation expressions dict contains both the validation (202) & 
    # test videos (305). so we simply load the test expressions dict and use it to filter out the test videos from
    # the validation expressions dict:
    test_meta_file = os.path.join(root, "meta_expressions", "test", "meta_expressions.json")
    with open(test_meta_file, 'r') as f:
        test_data = json.load(f)['videos']
    test_videos = set(test_data.keys())
    # valid_videos = valid_test_videos - test_videos
    # video_list = sorted([video for video in test_videos])
    video_list = sorted([video for video in test_videos])
    assert len(video_list) == 305, 'error: incorrect number of validation videos'

    start_time = time.time()
    print('Start inference')

    # sub_processor(args, test_data, img_folder, video_list)
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

    model.eval()
    tmp_dict = {}

    # 1. For each video
    for video in tqdm(video_list):
        metas = []  # list[dict], length is number of expressions
        tmp_dict[video] = {}
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
            tmp_dict[video][exp_id] = {}
            frames = meta[i]["frames"]

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
            with torch.cuda.amp.autocast(enabled=False):
                with torch.no_grad():
                    outputs = model([imgs], [exp], [target])
            pred_logits = outputs["pred_logits"][0] 
            pred_masks = outputs["pred_masks"][0]
            # according to pred_logits, select the query index
            tmp = pred_logits.sigmoid()  # [t, q, k]
            pred_scores = tmp.mean(0)    # [q, k]

            max_scores, _ = pred_scores.max(-1)  # [q,]
            _, max_ind = max_scores.max(-1)     # [1,]
            max_inds = max_ind.repeat(video_len)

            class_info = tmp[range(video_len), max_inds, ...].tolist()
            class_info = [cl[0] for cl in class_info]

            pred_masks = pred_masks[range(video_len), max_inds, ...] # [t, h, w]
            pred_masks = pred_masks.unsqueeze(0)

            pred_masks = F.interpolate(pred_masks, size=(origin_h, origin_w), mode='bilinear', align_corners=False) 
            # store the video results
            tmp_info = {
                'mask_logits': pred_masks.squeeze(0).cpu(),
                'class_info': class_info
            }
            tmp_dict[video][exp_id].update(tmp_info)
    # save tmp results

    if not os.path.exists(args.ensemble_save_path):
        os.makedirs(args.ensemble_save_path)
    save_path = os.path.join(args.ensembe_save_path, args.version)
    torch.save(tmp_dict, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MUTR inference script', parents=[opts.get_args_parser()])
    parser.add_argument("--version", type=str)
    args = parser.parse_args()
    main(args)
