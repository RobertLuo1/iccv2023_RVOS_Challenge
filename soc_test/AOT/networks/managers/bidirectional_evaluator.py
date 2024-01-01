import os
import time
import datetime as datetime
import json
import pickle
import shutil
from PIL import Image
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms


from dataloaders.eval_datasets import YOUTUBEVOS_Postprocess_Test, YOUTUBEVOS_Test, YOUTUBEVOS_DenseTest, DAVIS_Test, EVAL_TEST
import dataloaders.video_transforms as tr

from utils.image import flip_tensor, save_mask
from utils.checkpoint import load_network
from utils.eval import zip_folder

from networks.models import build_vos_model
from networks.engines import build_engine


class Evaluator(object):
    def __init__(self, cfg, rank=0, seq_queue=None, info_queue=None):
        self.gpu = cfg.TEST_GPU_ID + rank
        self.gpu_num = cfg.TEST_GPU_NUM
        self.rank = rank
        self.cfg = cfg
        self.seq_queue = seq_queue
        self.info_queue = info_queue

        self.print_log("Exp {}:".format(cfg.EXP_NAME))
        self.print_log(json.dumps(cfg.__dict__, indent=4, sort_keys=True))

        print("Use GPU {} for evaluating.".format(self.gpu))
        torch.cuda.set_device(self.gpu)

        self.print_log('Build VOS model.')
        self.model = build_vos_model(cfg.MODEL_VOS, cfg).cuda(self.gpu)

        self.process_pretrained_model()

        self.prepare_dataset()

    def process_pretrained_model(self):
        cfg = self.cfg

        if cfg.TEST_CKPT_PATH == 'test':
            self.ckpt = 'test'
            self.print_log('Test evaluation.')
            return

        if cfg.TEST_CKPT_PATH is None:
            if cfg.TEST_CKPT_STEP is not None:
                ckpt = str(cfg.TEST_CKPT_STEP)
            else:
                ckpts = os.listdir(cfg.DIR_CKPT)
                if len(ckpts) > 0:
                    ckpts = list(
                        map(lambda x: int(x.split('_')[-1].split('.')[0]),
                            ckpts))
                    ckpt = np.sort(ckpts)[-1]
                else:
                    self.print_log('No checkpoint in {}.'.format(cfg.DIR_CKPT))
                    exit()
            self.ckpt = ckpt
            if cfg.TEST_EMA:
                cfg.DIR_CKPT = os.path.join(cfg.DIR_RESULT, 'ema_ckpt')
            cfg.TEST_CKPT_PATH = os.path.join(cfg.DIR_CKPT,
                                              'save_step_%s.pth' % ckpt)
            self.model, removed_dict = load_network(self.model,
                                                    cfg.TEST_CKPT_PATH,
                                                    self.gpu)
            if len(removed_dict) > 0:
                self.print_log(
                    'Remove {} from pretrained model.'.format(removed_dict))
            self.print_log('Load latest checkpoint from {}'.format(
                cfg.TEST_CKPT_PATH))
        else:
            self.ckpt = 'unknown'
            self.model, removed_dict = load_network(self.model,
                                                    cfg.TEST_CKPT_PATH,
                                                    self.gpu)
            if len(removed_dict) > 0:
                self.print_log(
                    'Remove {} from pretrained model.'.format(removed_dict))
            self.print_log('Load checkpoint from {}'.format(
                cfg.TEST_CKPT_PATH))

    def prepare_dataset(self):
        cfg = self.cfg
        self.print_log('Process dataset...')
        eval_transforms = transforms.Compose([
            tr.MultiRestrictSize(cfg.TEST_MIN_SIZE, cfg.TEST_MAX_SIZE,
                                 cfg.TEST_FLIP, cfg.TEST_MULTISCALE,
                                 cfg.MODEL_ALIGN_CORNERS),
            tr.MultiToTensor()
        ])

        eval_name = '{}_{}_{}_{}_ckpt_{}'.format(cfg.TEST_DATASET,
                                                 cfg.TEST_DATASET_SPLIT,
                                                 cfg.EXP_NAME, cfg.STAGE_NAME,
                                                 self.ckpt)

        if cfg.TEST_EMA:
            eval_name += '_ema'
        if cfg.TEST_FLIP:
            eval_name += '_flip'
        if len(cfg.TEST_MULTISCALE) > 1:
            eval_name += '_ms_' + str(cfg.TEST_MULTISCALE).replace(
                '.', 'dot').replace('[', '').replace(']', '').replace(
                    ', ', '_')

        if 'youtubevos' in cfg.TEST_DATASET:
            self.result_root = os.path.join(cfg.DIR_EVALUATION,
                                            cfg.TEST_DATASET, eval_name,
                                            'Annotations')
            if '_all_frames' in cfg.TEST_DATASET_SPLIT:
                split = cfg.TEST_DATASET_SPLIT.split('_')[0]
                youtubevos_test = YOUTUBEVOS_DenseTest

                self.result_root_sparse = os.path.join(cfg.DIR_EVALUATION,
                                                       cfg.TEST_DATASET,
                                                       eval_name + '_sparse',
                                                       'Annotations')
                self.zip_dir_sparse = os.path.join(
                    cfg.DIR_EVALUATION, cfg.TEST_DATASET,
                    '{}_sparse.zip'.format(eval_name))
            elif 'post' in cfg.TEST_DATASET:
                youtubevos_test = YOUTUBEVOS_Postprocess_Test
            else:
                split = cfg.TEST_DATASET_SPLIT
                youtubevos_test = YOUTUBEVOS_Test

            self.result_root = os.path.join(cfg.DIR_YTB_POST_EVAL, 'Annotations_AOT_class_index')
            self.key_frames_id = cfg.PICKED_KEY_FRAMES_ID

            self.dataset = youtubevos_test(root=cfg.DIR_YTB_POST_EVAL,
                                           images_root=cfg.DIR_IMAGE,
                                           transform=eval_transforms,
                                           result_root=self.result_root,
                                           key_frames_id=cfg.PICKED_KEY_FRAMES_ID,
                                           split=cfg.TEST_PHRASE)

        elif cfg.TEST_DATASET == 'davis2017':
            resolution = 'Full-Resolution' if cfg.TEST_DATASET_FULL_RESOLUTION else '480p'
            self.result_root = os.path.join(cfg.DIR_EVALUATION,
                                            cfg.TEST_DATASET, eval_name,
                                            'Annotations', resolution)
            self.dataset = DAVIS_Test(
                split=[cfg.TEST_DATASET_SPLIT],
                root=cfg.DIR_DAVIS,
                year=2017,
                transform=eval_transforms,
                full_resolution=cfg.TEST_DATASET_FULL_RESOLUTION,
                result_root=self.result_root)

        elif cfg.TEST_DATASET == 'davis2016':
            resolution = 'Full-Resolution' if cfg.TEST_DATASET_FULL_RESOLUTION else '480p'
            self.result_root = os.path.join(cfg.DIR_EVALUATION,
                                            cfg.TEST_DATASET, eval_name,
                                            'Annotations', resolution)
            self.dataset = DAVIS_Test(
                split=[cfg.TEST_DATASET_SPLIT],
                root=cfg.DIR_DAVIS,
                year=2016,
                transform=eval_transforms,
                full_resolution=cfg.TEST_DATASET_FULL_RESOLUTION,
                result_root=self.result_root)

        elif cfg.TEST_DATASET == 'test':
            self.result_root = os.path.join(cfg.DIR_EVALUATION,
                                            cfg.TEST_DATASET, eval_name,
                                            'Annotations')
            self.dataset = EVAL_TEST(eval_transforms, self.result_root)

        else:
            self.print_log('Unknown dataset!')
            exit()

        self.print_log('Eval {} on {} {}:'.format(cfg.EXP_NAME,
                                                  cfg.TEST_DATASET,
                                                  cfg.TEST_DATASET_SPLIT))
        self.source_folder = os.path.join(cfg.DIR_EVALUATION, cfg.TEST_DATASET,
                                          eval_name, 'Annotations')
        self.zip_dir = os.path.join(cfg.DIR_EVALUATION, cfg.TEST_DATASET,
                                    '{}.zip'.format(eval_name))
        if not os.path.exists(self.result_root):
            try:
                os.makedirs(self.result_root)
            except Exception as inst:
                self.print_log(inst)
                self.print_log('Failed to mask dir: {}.'.format(
                    self.result_root))
        self.print_log('Done!')

    def evaluating(self):
        cfg = self.cfg
        self.model.eval()
        video_num = 0
        processed_video_num = 0
        total_time = 0
        total_frame = 0
        total_sfps = 0
        total_video_num = len(self.dataset)
        start_eval_time = time.time()

        if self.seq_queue is not None:
            if self.rank == 0:
                for seq_idx in range(total_video_num):
                    self.seq_queue.put(seq_idx)
                for _ in range(self.gpu_num):
                    self.seq_queue.put('END')
            coming_seq_idx = self.seq_queue.get()
        all_engines = []
        mask_preds = dict()
        with torch.no_grad():
            for seq_idx, seq_dataset in tqdm(enumerate(self.dataset)):
                if seq_dataset == None:
                    break
                video_num += 1
                seq_name = seq_dataset.seq_name
                exp_id = seq_dataset.exp_id
                frames = seq_dataset.frames
                if self.seq_queue is not None:
                    if coming_seq_idx == 'END':
                        break
                    elif coming_seq_idx != seq_idx:
                        continue
                    else:
                        coming_seq_idx = self.seq_queue.get()

                processed_video_num += 1

                for engine in all_engines:
                    engine.restart_engine()

                seq_name = seq_dataset.seq_name
                print('GPU {} - Processing Seq {}_{} [{}/{}]:'.format(
                    self.gpu, seq_name, exp_id, video_num, total_video_num))
                torch.cuda.empty_cache()

                seq_dataloader = DataLoader(seq_dataset,
                                            batch_size=1,
                                            shuffle=False,
                                            num_workers=cfg.TEST_WORKERS,
                                            pin_memory=True)


                # seq_total_time = 0
                # seq_total_frame = 0
                # seq_timers = []
                samples_list = []
                for frame_idx, samples in enumerate(seq_dataloader):
                    samples_list.append(samples)
                sample_length = len(samples_list)

                k = seq_dataset.key_frames_num
                save_path = os.path.join(self.result_root, seq_name, exp_id)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)      
                forward_pred_mask = None
                backward_pred_mask  = None

                refer_mask = samples_list[k][0]['current_label']

                # 1 x H x W --> obj_num x H x W
                obj_num = torch.unique(refer_mask)[-1]
                new_refer_mask = []
                for num in range(1, obj_num + 1):
                    temp_mask = torch.zeros_like(refer_mask[0][0], dtype=torch.float32)
                    temp_mask[refer_mask[0][0] == num] = 1.0
                    new_refer_mask.append(temp_mask)
                try:
                    new_refer_mask = torch.stack(new_refer_mask, dim=0)
                except:
                    print('Wrong mask.................. {}_{}'.format(seq_name, exp_id))
                    for name in frames:
                        shutil.copy(os.path.join(seq_dataset.label_root, name + ".png"), os.path.join(save_path, name + ".png"))
                    continue
                seq_mask = [new_refer_mask]

                if k < sample_length - 1:
                    forward_pred_mask = self.forward(samples_list[k:], processed_video_num, seq_name, k, 
                                    total_time, total_frame, total_sfps, obj_num, cfg, all_engines)
                if k > 0:
                    backward_pred_mask = self.forward(samples_list[:k + 1][::-1], processed_video_num, seq_name, k, 
                                    total_time, total_frame, total_sfps, obj_num, cfg, all_engines, stage='Backward')
                # seq_mask = forward_pred_mask + [refer_mask] + backward_pred_mask
                if forward_pred_mask:
                    seq_mask = seq_mask + forward_pred_mask
                if backward_pred_mask:
                    seq_mask = backward_pred_mask[::-1] + seq_mask




                for j in range(sample_length):
                        if j == k:
                            continue
                        frame_name = frames[j]
                        seq_mask_j = torch.argmax(seq_mask[j], dim=1)
                        mask = seq_mask_j.cpu().numpy().astype(np.float32)[0]
                        mask = Image.fromarray(mask * 255).convert('L')
                        save_file = os.path.join(save_path, frame_name + ".png")
                        mask.save(save_file)

                ## ensemble info save
            #     save_name_key = '{}_{}'.format(seq_name, exp_id)
            #     mask_preds[save_name_key] = []
            #     for j in range(sample_length):
            #         if j == k:
            #             mask = seq_mask[j].cpu().numpy().astype(np.float16)[0]
            #             mask_preds[save_name_key].append(mask)
            #             continue
            #         frame_name = frames[j]
            #         # print(seq_mask[j].shape)
            #         # seq_mask_j = torch.argmax(seq_mask[j], dim=1)
            #         mask = seq_mask[j].cpu().numpy().astype(np.float16)[0]
            #         mask_preds[save_name_key].append(mask)
            #     if (seq_idx + 1) % 10 == 0:
            #         mask_pred_save_path = os.path.join(self.result_root, '{}.pkl'.format(seq_idx))
            #         with open(mask_pred_save_path, 'wb') as f:
            #             pickle.dump(mask_preds, f, protocol=pickle.HIGHEST_PROTOCOL)
            #         mask_preds = dict()
            #     if seq_idx == 833:
            #         break
            # mask_pred_save_path = os.path.join(self.result_root, '{}.pkl'.format(seq_idx))
            # with open(mask_pred_save_path, 'wb') as f:
            #     pickle.dump(mask_preds, f, protocol=pickle.HIGHEST_PROTOCOL)


    def forward(self, sample_list, processed_video_num, seq_name, 
                     key_frame_idx, total_time, 
                     total_frame, total_sfps,
                     obj_num, cfg, all_engines, stage='Forward'):


        seq_total_time = 0
        seq_total_frame = 0
        seq_timers = []

        pred_masks = []
        with torch.no_grad():
            for frame_idx, samples in enumerate(sample_list):
                all_preds = []
                new_obj_label = None
                for aug_idx in range(len(samples)):
                    if len(all_engines) <= aug_idx:
                        all_engines.append(
                            build_engine(cfg.MODEL_ENGINE,
                                            phase='eval',
                                            aot_model=self.model,
                                            gpu_id=self.gpu,
                                            long_term_mem_gap=self.cfg.
                                            TEST_LONG_TERM_MEM_GAP))
                        all_engines[-1].eval()

                    engine = all_engines[aug_idx]

                    sample = samples[aug_idx]

                    is_flipped = sample['meta']['flip']

                    obj_nums = sample['meta']['obj_num']
                    imgname = sample['meta']['current_name']
                    ori_height = sample['meta']['height']
                    ori_width = sample['meta']['width']
                    # obj_idx = sample['meta']['obj_idx']

                    obj_nums = [int(obj_num) for obj_num in obj_nums]
                    # obj_idx = [int(_obj_idx) for _obj_idx in obj_idx]

                    current_img = sample['current_img']
                    current_img = current_img.cuda(self.gpu,
                                                    non_blocking=True)
                    sample['current_img'] = current_img
    
                    if 'current_label' in sample.keys():
                        current_label = sample['current_label'].cuda(
                            self.gpu, non_blocking=True).float()
                    else:
                        current_label = None

                    #############################################################

                    if frame_idx == 0:
                        _current_label = F.interpolate(
                            current_label,
                            size=current_img.size()[2:],
                            mode="nearest")
                        # import pdb; pdb.set_trace()
                        engine.add_reference_frame(current_img,
                                                    _current_label,
                                                    frame_step=0,
                                                    obj_nums=obj_nums)
                    else:
                        if aug_idx == 0:
                            seq_timers.append([])
                            now_timer = torch.cuda.Event(
                                enable_timing=True)
                            now_timer.record()
                            seq_timers[-1].append((now_timer))
                        engine.match_propogate_one_frame(current_img)
                        pred_logit = engine.decode_current_logits(
                            (ori_height, ori_width))
                        
                        if is_flipped:
                            pred_logit = flip_tensor(pred_logit, 3)

                        pred_prob = torch.softmax(pred_logit, dim=1)
                        all_preds.append(pred_prob)

                        if not is_flipped and current_label is not None and new_obj_label is None:
                            new_obj_label = current_label
                
                if frame_idx > 0:
                    # import pdb; pdb.set_trace()
                    all_preds = torch.cat(all_preds, dim=0)
                    pred_prob = torch.mean(all_preds, dim=0, keepdim=True)
                    pred_label = torch.argmax(pred_prob,
                                                dim=1,
                                                keepdim=True).float()
                    pred_masks.append(pred_prob)
                    # import pdb; pdb.set_trace()
                    if new_obj_label is not None:
                        keep = (new_obj_label == 0).float()
                        pred_label = pred_label * \
                            keep + new_obj_label * (1 - keep)
                        new_obj_nums = [int(pred_label.max().item())]

                        if cfg.TEST_FLIP:
                            flip_pred_label = flip_tensor(pred_label, 3)

                        for aug_idx in range(len(samples)):
                            engine = all_engines[aug_idx]
                            current_img = samples[aug_idx]['current_img']

                            current_label = flip_pred_label if samples[
                                aug_idx]['meta']['flip'] else pred_label
                            current_label = F.interpolate(
                                current_label,
                                size=engine.input_size_2d,
                                mode="nearest")
                            engine.add_reference_frame(
                                current_img,
                                current_label,
                                obj_nums=new_obj_nums,
                                frame_step=frame_idx)
                    else:
                        if not cfg.MODEL_USE_PREV_PROB:
                            if cfg.TEST_FLIP:
                                flip_pred_label = flip_tensor(
                                    pred_label, 3)

                            for aug_idx in range(len(samples)):
                                engine = all_engines[aug_idx]
                                current_label = flip_pred_label if samples[
                                    aug_idx]['meta']['flip'] else pred_label
                                current_label = F.interpolate(
                                    current_label,
                                    size=engine.input_size_2d,
                                    mode="nearest")
                                engine.update_memory(current_label)
                        else:
                            if cfg.TEST_FLIP:
                                flip_pred_prob = flip_tensor(pred_prob, 3)

                            for aug_idx in range(len(samples)):
                                engine = all_engines[aug_idx]
                                current_prob = flip_pred_prob if samples[
                                    aug_idx]['meta']['flip'] else pred_prob
                                current_prob = F.interpolate(
                                    current_prob,
                                    size=engine.input_size_2d,
                                    mode="nearest")
                                engine.update_memory(current_prob)


        return pred_masks


    def print_log(self, string):
        if self.rank == 0:
            print(string)
