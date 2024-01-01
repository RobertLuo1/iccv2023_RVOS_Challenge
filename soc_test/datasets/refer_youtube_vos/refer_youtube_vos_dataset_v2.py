import os
import json
import torch
import random
import datasets.refer_youtube_vos.transforms.transform_video as T
from einops import rearrange
import misc as utils
from pathlib import Path

import torch
from torch.autograd.grad_mode import F
from torch.utils.data import Dataset
# import datasets.transforms_video as T

import os
from PIL import Image
import json
import numpy as np
import random

# from datasets.categories import ytvos_category_dict as category_dict


    # with open('train_meta_num_frames_{}.json'.format(num_frames), 'w') as f:
    #     json.dump(metas, f, indent=2)

"""
Ref-YoutubeVOS data loader
"""
class YTVOSDataset(Dataset):
    """
    A dataset class for the Refer-Youtube-VOS dataset which was first introduced in the paper:
    "URVOS: Unified Referring Video Object Segmentation Network with a Large-Scale Benchmark"
    (see https://link.springer.com/content/pdf/10.1007/978-3-030-58555-6_13.pdf).
    The original release of the dataset contained both 'first-frame' and 'full-video' expressions. However, the first
    dataset is not publicly available anymore as now only the harder 'full-video' subset is available to download
    through the Youtube-VOS referring video object segmentation competition page at:
    https://competitions.codalab.org/competitions/29139
    Furthermore, for the competition the subset's original validation set, which consists of 507 videos, was split into
    two competition 'validation' & 'test' subsets, consisting of 202 and 305 videos respectively. Evaluation can
    currently only be done on the competition 'validation' subset using the competition's server, as
    annotations were publicly released only for the 'train' subset of the competition.

    """
    def __init__(self, sub_type:str, img_folder: Path, ann_file: Path,
                 window_size: int, **kwargs):
        self.sub_type = sub_type
        if self.sub_type == 'test':
            self.sub_type = 'valid'
        self.img_folder = os.path.join(img_folder, self.sub_type)
        self.ann_file = os.path.join(ann_file, self.sub_type, 'meta_expressions.json')           
        self.num_frames = window_size     

        self._transforms = make_coco_transforms(self.sub_type, max_size=640)
        self.collator = Collator(self.sub_type)
        
        # create video meta data
        if self.sub_type == 'train':
            self.prepare_metas()
            # metadata_file_path = './train_meta_num_frames_8.json'
            # with open(metadata_file_path, 'r') as f:
            #     self.metas = json.load(f)       
        else:
            metadata_file_path = f'./datasets/refer_youtube_vos/valid_samples_metadata.json'
            with open(metadata_file_path, 'r') as f:
                self.metas = [tuple(a) for a in json.load(f)]
        # print('\n video num: ', len(self.videos), ' clip num: ', len(self.metas))  
        # print('\n')    
    def prepare_metas(self):
        # img_folder = '/mnt/data_16TB/lzy23/rvosdata/refer_youtube_vos/train'
        # ann_file = '/mnt/data_16TB/lzy23/rvosdata/refer_youtube_vos/meta_expressions/train/meta_expressions.json'
        
        # read object information
        with open(os.path.join(str(self.img_folder), 'meta.json'), 'r') as f:
            subset_metas_by_video = json.load(f)['videos']
        
        # read expression data
        with open(str(self.ann_file), 'r') as f:
            subset_expressions_by_video = json.load(f)['videos'] #meta_expression.json
            videos = list(subset_expressions_by_video.keys())

        self.metas = []
        for vid in videos:
            vid_meta = subset_metas_by_video[vid]
            vid_data = subset_expressions_by_video[vid]
            vid_frames = sorted(vid_data['frames'])
            vid_len = len(vid_frames)
            for exp_id, exp_dict in vid_data['expressions'].items():
                for frame_id in range(0, vid_len, self.num_frames):
                    meta = {}
                    meta['video'] = vid
                    meta['exp'] = exp_dict['exp']
                    meta['obj_id'] = int(exp_dict['obj_id'])
                    meta['frames'] = vid_frames
                    meta['frame_id'] = frame_id
                    # get object category
                    obj_id = exp_dict['obj_id']
                    meta['category'] = vid_meta['objects'][obj_id]['category']
                    self.metas.append(meta)
    @staticmethod
    def bounding_box(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return rmin, rmax, cmin, cmax # y1, y2, x1, x2 
        
    def __len__(self):
        return len(self.metas)
        
    def __getitem__(self, idx):
        if self.sub_type == 'train':
            instance_check = False
            while not instance_check:
                meta = self.metas[idx]  # dict

                video, exp, obj_id, category, frames, frame_id = \
                            meta['video'], meta['exp'], meta['obj_id'], meta['category'], meta['frames'], meta['frame_id']
                # clean up the caption
                exp = " ".join(exp.lower().split())
                # category_id = category_dict[category]
                vid_len = len(frames)

                num_frames = self.num_frames
                # random sparse sample
                sample_indx = [frame_id]
                if self.num_frames != 1:
                    # local sample
                    sample_id_before = random.randint(1, 3)
                    sample_id_after = random.randint(1, 3)
                    local_indx = [max(0, frame_id - sample_id_before), min(vid_len - 1, frame_id + sample_id_after)]
                    sample_indx.extend(local_indx)
        
                    # global sampling
                    if num_frames > 3:
                        all_inds = list(range(vid_len))
                        global_inds = all_inds[:min(sample_indx)] + all_inds[max(sample_indx):]
                        global_n = num_frames - len(sample_indx)
                        if len(global_inds) > global_n:
                            select_id = random.sample(range(len(global_inds)), global_n)
                            for s_id in select_id:
                                sample_indx.append(global_inds[s_id])
                        elif vid_len >=global_n:  # sample long range global frames
                            select_id = random.sample(range(vid_len), global_n)
                            for s_id in select_id:
                                sample_indx.append(all_inds[s_id])
                        else:
                            select_id = random.sample(range(vid_len), global_n - vid_len) + list(range(vid_len))           
                            for s_id in select_id:                                                                   
                                sample_indx.append(all_inds[s_id])
                sample_indx.sort()

                # read frames and masks
                imgs, masks, = [], []
                for j in range(self.num_frames):
                    frame_indx = sample_indx[j]
                    frame_name = frames[frame_indx]
                    img_path = os.path.join(str(self.img_folder), 'JPEGImages', video, frame_name + '.jpg')
                    mask_path = os.path.join(str(self.img_folder), 'Annotations', video, frame_name + '.png')
                    img = Image.open(img_path).convert('RGB')
                    mask = torch.tensor(np.array(Image.open(mask_path)))

                    # create the target
                    # label =  torch.tensor(category_id) 
                    # mask = np.array(mask)
                    # mask = (mask==obj_id).astype(np.float32) # 0,1 binary
                    # if (mask > 0).any():
                    #     y1, y2, x1, x2 = self.bounding_box(mask)
                    #     box = torch.tensor([x1, y1, x2, y2]).to(torch.float)
                    #     valid.append(1)
                    # else: # some frame didn't contain the instance
                    #     box = torch.tensor([0, 0, 0, 0]).to(torch.float) 
                    #     valid.append(0)
                    # mask = torch.from_numpy(mask)

                    # append
                    imgs.append(img)
                    # labels.append(label)
                    masks.append(mask)
                    # boxes.append(box)

                # transform
                # w, h = img.size
                # labels = torch.stack(labels, dim=0) 
                # boxes = torch.stack(boxes, dim=0) 
                # boxes[:, 0::2].clamp_(min=0, max=w)
                # boxes[:, 1::2].clamp_(min=0, max=h)
                all_object_indices = set().union(*[m.unique().tolist() for m in masks])
                all_object_indices.remove(0)  # remove the background index
                all_object_indices = sorted(list(all_object_indices))
                mask_annotations_by_object = []
                # masks = torch.stack(masks, dim=0)
                for obj_id in all_object_indices:
                    obj_id_mask_annotations = torch.stack([(m == obj_id).to(torch.uint8) for m in masks])
                    mask_annotations_by_object.append(obj_id_mask_annotations)
                mask_annotations_by_object = torch.stack(mask_annotations_by_object)
                mask_annotations_by_frame = rearrange(mask_annotations_by_object, 'o t h w -> t o h w')  # o for object 
                ref_obj_idx = torch.tensor(all_object_indices.index(obj_id), dtype=torch.long)
                targets = []
                for frame_masks in mask_annotations_by_frame:
                    target = {'masks': frame_masks,
                            # idx in 'masks' of the text referred instance
                            'referred_instance_idx': ref_obj_idx,
                            # whether the referred instance is visible in the frame:
                            'is_ref_inst_visible': frame_masks[ref_obj_idx].any(),
                            'orig_size': frame_masks.shape[-2:],  # original frame shape without any augmentations
                            # size with augmentations, will be changed inside transforms if necessary
                            'size': frame_masks.shape[-2:],
                            'iscrowd': torch.zeros(len(frame_masks)),  # for compatibility with DETR COCO transforms
                            'caption': exp,
                            }
                    targets.append(target)
                # target = {
                #     #'frames_idx': torch.tensor(sample_indx), # [T,]
                #     #'labels': labels,                        # [T,]
                #     # 'boxes': boxes,                          # [T, 4], xyxy
                #     'referred_instance_idx': ref_obj_idx,
                #     'masks': masks,                          # [T, H, W]
                #     # 'valid': torch.tensor(valid),            # [T,]
                #     'caption': exp,
                #     'orig_size': torch.as_tensor([int(h), int(w)]), 
                #     'size': torch.as_tensor([int(h), int(w)])
                # }

                # "boxes" normalize to [0, 1] and transform from xyxy to cxcywh in self._transform
                # for idx, (img, target) in enumerate(zip(imgs, targets)):
                imgs, targets = self._transforms(imgs, targets)
                    # imgs[idx] = img
                    # targets[idx] = target 
                imgs = torch.stack(imgs, dim=0) # [T, 3, H, W]
                
                # FIXME: handle "valid", since some box may be removed due to random crop
                for t in targets:
                    if bool(t['is_ref_inst_visible'].item()):
                        instance_check = True
                        break
                if not instance_check: 
                    idx = random.randint(0, self.__len__() - 1)
                # if torch.any(target['valid'] == 1):  # at leatst one instance
                #     instance_check = True
                # else:
                #     idx = random.randint(0, self.__len__() - 1)

            return imgs, targets, targets[0]['caption']
        else:
            video_id, frame_indices, text_query_dict = self.metas[idx]
            text_query = text_query_dict['exp']
            text_query = " ".join(text_query.lower().split())
            imgs_path = [os.path.join(self.img_folder, 'JPEGImages', video_id, f'{idx}.jpg') for idx in frame_indices]
            imgs = [Image.open(img_path) for img_path in imgs_path]
            original_frame_size = imgs[0].size[::-1] # H W
            targets = len(imgs) * [None]
            # for idx, (img, target) in enumerate(zip(imgs, targets)):
            imgs, targets = self._transforms(imgs, targets)
                # imgs[idx] = img
                # targets[idx] = target 
            imgs = torch.stack(imgs, dim=0)
            video_metadata = {'video_id': video_id,
                              'frame_indices': frame_indices,
                              'resized_frame_size': imgs[0].shape[-2:],
                              'original_frame_size': original_frame_size,
                              'exp_id': text_query_dict['exp_id']}
            return imgs, video_metadata, text_query


def make_coco_transforms(image_set, max_size=640):
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [288, 320, 352, 392, 416, 448, 480, 512]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.PhotometricDistort(),
            T.RandomSelect(
                T.Compose([
                    T.RandomResize(scales, max_size=max_size), #current modify here
                    T.Check(),
                ]),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=max_size),
                    T.Check(),
                ])
            ),
            normalize,
        ])
    
    # we do not use the 'val' set since the annotations are inaccessible
    if image_set == 'valid':
        return T.Compose([
            T.RandomResize([360], max_size=640),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

class Collator:
    def __init__(self, subset_type):
        self.subset_type = subset_type

    def __call__(self, batch):
        if self.subset_type == 'train':
            samples, targets, text_queries = list(zip(*batch))
            samples = utils.nested_tensor_from_videos_list(samples)  # [T, B, C, H, W]
            # convert targets to a list of tuples. outer list - time steps, inner tuples - time step batch
            targets = list(zip(*targets))
            batch_dict = {
                'samples': samples,
                'targets': targets,
                'text_queries': text_queries
            }
            return batch_dict
        else:  # validation:
            samples, videos_metadata, text_queries = list(zip(*batch))
            samples = utils.nested_tensor_from_videos_list(samples)  # [T, B, C, H, W]
            batch_dict = {
                'samples': samples,
                'videos_metadata': videos_metadata,
                'text_queries': text_queries
            }
            return batch_dict