"""
This file contains a Trainer class which handles the training and evaluation of MTTR.
"""
import math
import sys
import os
from os import path
import shutil
import random
import numpy as np
import wandb
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.cuda.amp as amp
from PIL import Image
from tqdm import tqdm
import gc
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from metrics import calculate_precision_at_k_and_iou_metrics
from utils import create_output_dir, create_checkpoint_dir, flatten_temporal_batch_dims, cosine_lr
from datasets import build_dataset
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import MultiStepLR, CyclicLR
import misc as utils
from models import build_model
from models.video_swin_transformer import compute_mask
import json
from functools import partial
from models.ema import ExponentialMovingAverage

def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

class Trainer:
    def __init__(self, config, process_id, device_id, num_processes):
        self.config = config

        self.world_size = num_processes
        self.distributed = num_processes > 1
        self.process_id = process_id
        self.is_main_process = process_id == 0
        self.device = init_process_group_and_set_device(num_processes, process_id, device_id, config)

        # fix the seed for reproducibility
        # seed = config.seed + config.rank
        seed = config.seed + config.rank
        torch.manual_seed(seed)
        # torch.cuda.manual_seed(seed)  
        # torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

        model, criterion, postprocessor = build_model(config)
        model.to(self.device)
        model_without_ddp = model
        if config.distributed:
            model = DDP(model, device_ids=[device_id])
            model_without_ddp = model.module
        self.model = model
        self.backbone_name = config.backbone
        self.criterion = criterion
        self.postprocessor = postprocessor

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)

        self.dataset_name = config.dataset_name
        if self.dataset_name == 'a2d_sentences' or self.dataset_name == 'jhmdb_sentences':
            self.evaluate = self.evaluate_a2d_sentences
            if self.config.use_ema:
                self.ema_evaluate = self.evaluate_a2d_sentences_ema
        elif self.dataset_name == 'ref_youtube_vos':
            self.evaluate = self.evaluate_refer_youtube_vos
            if self.config.use_ema:
                self.ema_evaluate = self.evaluate_refer_youtube_vos_ema
        else:
            assert False, f'error: dataset {self.dataset_name} is not supported'

        dataset_train = build_dataset(image_set='train', dataset_file=self.dataset_name, **vars(config))
        dataset_val = build_dataset(image_set='test', dataset_file=self.dataset_name, **vars(config))
        if self.distributed:
            self.sampler_train = DistributedSampler(dataset_train, num_replicas=config.world_size, rank=config.rank,
                                                    shuffle=True, seed=config.seed, drop_last=False)
        else:
            self.sampler_train = None
        self.data_loader_train = DataLoader(dataset_train, batch_size=config.batch_size, sampler=self.sampler_train,
                                            collate_fn=dataset_train.collator, num_workers=config.num_workers,
                                            pin_memory=True, shuffle=self.sampler_train is None)
        # self.data_loader_train = DataLoader(dataset_train, batch_size=config.batch_size, sampler=self.sampler_train,
        #                                     collate_fn=dataset_train.collator, num_workers=config.num_workers,
        #                                     pin_memory=True, shuffle=self.sampler_train is None, worker_init_fn=partial(
        #                                     worker_init_fn, num_workers=config.num_workers, rank=config.rank,
        #                                     seed=seed) if seed is not None else None,)
        if self.distributed:
            sampler_val = DistributedSampler(dataset_val, num_replicas=config.world_size, rank=config.rank, shuffle=False)
        else:
            sampler_val = None
        eval_batch_size = config.eval_batch_size
        self.data_loader_val = DataLoader(dataset_val, eval_batch_size, sampler=sampler_val, drop_last=False,
                                          collate_fn=dataset_val.collator, num_workers=config.num_workers,
                                          pin_memory=True)

        # Optimizer, LR-Scheduler, AMP Grad Scaler:
        param_dicts = [
            {"params": [p for n, p in model_without_ddp.named_parameters()
                        if "backbone" not in n and "text_encoder" not in n and p.requires_grad]},
            {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
             "lr": config.lr_backbone},
            {"params": [p for n, p in model_without_ddp.named_parameters() if "text_encoder" in n and p.requires_grad],
             "lr": config.text_encoder_lr},
        ]
        self.optimizer = torch.optim.AdamW(param_dicts, lr=config.lr, weight_decay=config.weight_decay)
        self.num_batches_per_epoch = len(self.data_loader_train)
        if self.dataset_name == 'a2d_sentences':
            if self.config.use_cycle_lr:
                self.lr_scheduler = CyclicLR(self.optimizer, base_lr=config.base_lr, max_lr=[config.lr, config.lr_backbone, config.text_encoder_lr], cycle_momentum=False)
            else:
                self.lr_scheduler = MultiStepLR(self.optimizer, milestones=config.lr_drop, gamma=0.2, verbose=True)
            # total_steps = self.num_batches_per_epoch * config.epochs
            # self.lr_scheduler = cosine_lr(self.optimizer, config.lr, 2000, total_steps)
        else:  # refer-youtube-vos:
            if self.config.use_cycle_lr:
                self.lr_scheduler = CyclicLR(self.optimizer, base_lr=config.base_lr, max_lr=[config.lr, config.lr_backbone, config.text_encoder_lr], cycle_momentum=False)
            else:
                self.lr_scheduler = MultiStepLR(self.optimizer, milestones=config.lr_drop, gamma=0.1, verbose=True)
        self.grad_scaler = amp.GradScaler(enabled=config.enable_amp)
        self.max_norm = config.clip_max_norm
        self.model_ema = None
        if config.use_ema:
        # Decay adjustment that aims to keep the decay independent of other hyper-parameters originally proposed at:
        # https://github.com/facebookresearch/pycls/blob/f8cd9627/pycls/core/net.py#L123
        #
        # total_ema_updates = (Dataset_size / n_GPUs) * epochs / (batch_size_per_gpu * EMA_steps)
        # We consider constant = Dataset_size for a given dataset/setup and omit it. Thus:
        # adjust = 1 / total_ema_updates ~= n_GPUs * batch_size_per_gpu * EMA_steps / epochs
            adjust = config.world_size * config.batch_size * config.model_ema_steps / config.epochs
            alpha = 1.0 - config.model_ema_decay
            alpha = min(1.0, alpha * adjust)
            self.model_ema = ExponentialMovingAverage(model_without_ddp, device=self.device, decay=1.0 - alpha)

        if self.is_main_process:
            self.output_dir_path = create_output_dir(config)
            self.checkpoint_dir_path = create_checkpoint_dir(self.output_dir_path)
            if config.wandb_mode == 'online':
                wandb.init(project='RefVOS', config=config, mode=config.wandb_mode, name='mttr_vita')
            print(config)
        else:
            self.output_dir_path = ''
        if self.distributed:
            # sync the newly created output dir among all processes:
            output_dir_sync_list = [None for _ in range(self.world_size)]
            dist.all_gather_object(output_dir_sync_list, self.output_dir_path)
            self.output_dir_path = output_dir_sync_list[0]

        self.total_epochs = config.epochs
        self.epoch = 0
        self.iteration = 0
        self.best_mAP = 0
        self.best_loss = math.inf

        if self.config.joint_finetune:
            print("============================================>")
            print("Load pretrained weights from {} ...".format(self.config.pretrained_weights))
            checkpoint_dict = torch.load(self.config.pretrained_weights, map_location="cpu")
            missing_keys, unexpected_keys = model_without_ddp.load_state_dict(checkpoint_dict['model_state_dict'], strict=False)
            unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
            if len(missing_keys) > 0:
                print('Missing Keys: {}'.format(missing_keys))
            if len(unexpected_keys) > 0:
                print('Unexpected Keys: {}'.format(unexpected_keys))

        elif self.dataset_name != "davis" and self.dataset_name != "jhmdb" and self.config.pretrained_weights is not None:
            print("============================================>")
            print("Load pretrained weights from {} ...".format(self.config.pretrained_weights))
            checkpoint = torch.load(self.config.pretrained_weights, map_location="cpu")
            checkpoint_dict = pre_trained_model_to_finetune(checkpoint, self.config)
            model_without_ddp.load_state_dict(checkpoint_dict, strict=False)
            print("============================================>")

        

    def train(self):
        print("Training started...")
        for self.epoch in tqdm(range(self.epoch, self.total_epochs), disable=not self.is_main_process):
            self.model.train()
            self.criterion.train()
            metric_logger = utils.MetricLogger(delimiter="  ")
            metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
            header = 'Epoch: [{}]'.format(self.epoch)
            print_freq = 10
            if self.distributed:
                self.sampler_train.set_epoch(self.epoch)
            total_epoch_loss = 0
            loss_sums_dict = {k: 0 for k in self.criterion.weight_dict.keys()}
            for index, batch_dict in enumerate(tqdm(self.data_loader_train, disable=not utils.is_main_process())):
                samples = batch_dict['samples'].to(self.device)
                targets = to_device(batch_dict['targets'], self.device)
                text_queries = batch_dict['text_queries']
                # keep only the valid targets (targets of frames which are annotated). for example, in a2d-sentences
                # only the center frame in each window is annotated.
                if self.config.dataset_name == 'a2d_sentences':
                    valid_indices = []
                    new_targets = []
                    frames = len(targets)
                    batch = len(targets[0])
                    for b in range(batch):
                        for i, t in enumerate(targets):
                            if targets[i][b] is not None:
                                valid_indices.append(i + b * frames)
                                new_targets.append(targets[i][b])
                    valid_indices = torch.tensor(valid_indices).to(self.device)
                    targets = [tuple(new_targets)]
                else:
                    valid_indices = None
                # else:
                # valid_indices = torch.tensor([i for i, t in enumerate(targets) if None not in t]).to(self.device)
                # targets = [targets[i] for i in valid_indices.tolist()] #List[(target_dict, target_dict)]

                with amp.autocast(enabled=self.config.enable_amp):
                    outputs = self.model(samples, valid_indices, text_queries, targets)
                    loss_dict = self.criterion(outputs, targets)
                    weight_dict = self.criterion.weight_dict
                    losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

                # reduce losses over all GPUs for logging purposes
                loss_dict_reduced = utils.reduce_dict(loss_dict)
                loss_dict_reduced_scaled = {k: v * weight_dict[k] for k, v in loss_dict_reduced.items() if
                                            k in weight_dict}
                total_loss_reduced = sum(loss_dict_reduced_scaled.values()).item()
                if not math.isfinite(total_loss_reduced):
                    print("Loss is {}, stopping training".format(total_loss_reduced))
                    print(loss_dict_reduced)
                    sys.exit(1)

                self.optimizer.zero_grad()
                self.grad_scaler.scale(losses).backward()
                if self.max_norm > 0:
                    self.grad_scaler.unscale_(self.optimizer)  # gradients must be unscaled before clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm, error_if_nonfinite=False)
                self.grad_scaler.step(self.optimizer)
                self.grad_scaler.update()

                metric_logger.update(loss=total_loss_reduced, **loss_dict_reduced_scaled,)
                metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

                # if self.is_main_process:
                #     wandb.log({'total_iteration_loss': total_loss_reduced})
                self.iteration += 1
                total_epoch_loss += total_loss_reduced
                for k in loss_sums_dict.keys():
                    loss_sums_dict[k] += loss_dict_reduced_scaled.get(k, torch.zeros(1)).item()

                if self.config.use_ema and index % self.config.model_ema_steps == 0:
                    self.model_ema.update_parameters(self.model)
                    if self.epoch < self.config.model_ema_start_epoch:
                        self.model_ema.n_averaged.fill_(0)
                if self.config.use_cycle_lr:
                    self.lr_scheduler.step()
                    if self.is_main_process:
                        if index % 500 == 0:
                            print()
                            print(self.optimizer.param_groups[0]["lr"])
                    # if self.is_main_process:
                        # count = 0
                        # ema_params = list(self.model_ema.parameters())
                        # model_params = list(self.model.parameters())
                        # number = len(ema_params)
                        # for p1, p2 in zip(ema_params, model_params):
                        #     if torch.allclose(p1.data, p2.data):
                        #         count +=1
                        # print("--------------------")
                        # print(number, count)
                    if self.epoch < self.config.model_ema_start_epoch:
                        self.model_ema.n_averaged.fill_(0)
                    #     print(self.model_ema.n_averaged)
                #use warmups
                # step = self.num_batches_per_epoch * self.epoch + i
                # self.lr_scheduler(step)
                
            
            metric_logger.synchronize_between_processes()
            train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': self.epoch}

            # if self.dataset_name == 'a2d_sentences':
            #     self.lr_scheduler.step()
            # else:  # refer-youtube-vos
            #     self.lr_scheduler.step(total_epoch_loss)  # note that this loss is synced across all processes
            if not self.config.use_cycle_lr:
                self.lr_scheduler.step()
            
            # evaluation:
            # run gc collection before starting evaluation to avoid possible OOM errors due to swin-T caching:
            self.clear_memory()
            if self.epoch >= 0:
                eval_metrics = self.evaluate()
                for key, value in eval_metrics.items():
                    log_stats['evaluate' + key] = value
                if self.model_ema:
                    ema_log_stats = {}
                    ema_eval_metrics = self.ema_evaluate()
                    for key, value in ema_eval_metrics.items():
                        ema_log_stats['evaluate' + key] = value
            if self.is_main_process:
                if self.dataset_name == 'a2d_sentences':
                    mAP_score = eval_metrics.get('mAP 0.5:0.95')
                    self.save_checkpoint(mAP_score)
                else:  # refer-youtube-vos:
                    self.save_checkpoint(total_epoch_loss)
                # eval_metrics.update({'epoch': self.epoch, 'epoch_loss': total_epoch_loss})
                # eval_metrics.update(loss_sums_dict)
                if self.config.wandb_mode == 'online':
                    wandb.log(log_stats)
                with open(os.path.join(self.output_dir_path,'log.txt'), 'a')as f:
                    f.write(json.dumps(log_stats) + "\n")
                if self.model_ema:
                    with open(os.path.join(self.output_dir_path,'log_ema.txt'), 'a')as f:
                        f.write(json.dumps(ema_log_stats) + "\n")
                # wandb.log({'main_model_learning_rate': self.optimizer.param_groups[0]['lr']})

            # run gc collection before starting a new epoch to avoid possible OOM errors due to swinT caching :
            self.clear_memory()
            if self.distributed:
                dist.barrier()
    
    @torch.no_grad()
    def evaluate_a2d_sentences(self):
        self.model.eval()
        predictions = []
        for batch_dict in tqdm(self.data_loader_val, disable=not self.is_main_process):
            samples = batch_dict['samples'].to(self.device)
            targets = to_device(batch_dict['targets'], self.device)
            text_queries = batch_dict['text_queries']

            # keep only the valid targets (targets of frames which are annotated):
            valid_indices = []
            new_targets = []
            frames = len(targets)
            for b in range(self.config.eval_batch_size):
                for i, t in enumerate(targets):
                    if targets[i][b] is not None:
                        valid_indices.append(i + b * frames)
                        new_targets.append(targets[i][b])
            valid_indices = torch.tensor(valid_indices).to(self.device)
            targets = [tuple(new_targets)]
            # valid_indices = torch.tensor([i for i, t in enumerate(targets) if None not in t]).to(self.device)
            # targets = [targets[i] for i in valid_indices.tolist()]

            outputs = self.model(samples, valid_indices, text_queries, targets)
            outputs.pop('aux_outputs', None)

            outputs, targets = flatten_temporal_batch_dims(outputs, targets)
            processed_outputs = self.postprocessor(outputs, resized_padded_sample_size=samples.tensors.shape[-2:],
                                                   resized_sample_sizes=[t['size'] for t in targets],
                                                   orig_sample_sizes=[t['orig_size'] for t in targets])
            image_ids = [t['image_id'] for t in targets]
            for p, image_id in zip(processed_outputs, image_ids):
                for s, m  in zip(p['scores'], p['rle_masks']):
                    predictions.append({'image_id': image_id,
                                        'category_id': 1,  # dummy label, as categories are not predicted in ref-vos
                                        'segmentation': m,
                                        'score': s.item()})

        if self.distributed:
            # gather and merge predictions from all processes:
            gathered_pred_lists = utils.all_gather(predictions)
            predictions = [p for p_list in gathered_pred_lists for p in p_list]
        eval_metrics = {}
        if self.is_main_process:
            coco_gt = COCO(self.config.dataset_coco_gt_format_path)
            coco_pred = coco_gt.loadRes(predictions)
            coco_eval = COCOeval(coco_gt, coco_pred, iouType='segm')
            coco_eval.params.useCats = 0  # ignore categories as they are not predicted in ref-vos task
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            ap_labels = ['mAP 0.5:0.95', 'AP 0.5', 'AP 0.75', 'AP 0.5:0.95 S', 'AP 0.5:0.95 M', 'AP 0.5:0.95 L']
            ap_metrics = coco_eval.stats[:6]
            eval_metrics = {l: m for l, m in zip(ap_labels, ap_metrics)}
            if self.config.calculate_precision_and_iou_metrics:
                precision_at_k, overall_iou, mean_iou = calculate_precision_at_k_and_iou_metrics(coco_gt, coco_pred)
                eval_metrics.update({f'P@{k}': m for k, m in zip([0.5, 0.6, 0.7, 0.8, 0.9], precision_at_k)})
                eval_metrics.update({'overall_iou': overall_iou, 'mean_iou': mean_iou})
            print(eval_metrics)
        if self.distributed:
            dist.barrier()  # sync all processes before starting a new epoch or exiting
        return eval_metrics

    @torch.no_grad()
    def evaluate_a2d_sentences_ema(self):
        self.model_ema.eval()
        predictions = []
        for batch_dict in tqdm(self.data_loader_val, disable=not self.is_main_process):
            samples = batch_dict['samples'].to(self.device)
            targets = to_device(batch_dict['targets'], self.device)
            text_queries = batch_dict['text_queries']

            # keep only the valid targets (targets of frames which are annotated):
            valid_indices = []
            new_targets = []
            frames = len(targets)
            for b in range(self.config.eval_batch_size):
                for i, t in enumerate(targets):
                    if targets[i][b] is not None:
                        valid_indices.append(i + b * frames)
                        new_targets.append(targets[i][b])
            valid_indices = torch.tensor(valid_indices).to(self.device)
            targets = [tuple(new_targets)]
            # valid_indices = torch.tensor([i for i, t in enumerate(targets) if None not in t]).to(self.device)
            # targets = [targets[i] for i in valid_indices.tolist()]

            outputs = self.model_ema(samples, valid_indices, text_queries, targets)
            outputs.pop('aux_outputs', None)

            outputs, targets = flatten_temporal_batch_dims(outputs, targets)
            processed_outputs = self.postprocessor(outputs, resized_padded_sample_size=samples.tensors.shape[-2:],
                                                   resized_sample_sizes=[t['size'] for t in targets],
                                                   orig_sample_sizes=[t['orig_size'] for t in targets])
            image_ids = [t['image_id'] for t in targets]
            for p, image_id in zip(processed_outputs, image_ids):
                for s, m  in zip(p['scores'], p['rle_masks']):
                    predictions.append({'image_id': image_id,
                                        'category_id': 1,  # dummy label, as categories are not predicted in ref-vos
                                        'segmentation': m,
                                        'score': s.item()})

        if self.distributed:
            # gather and merge predictions from all processes:
            gathered_pred_lists = utils.all_gather(predictions)
            predictions = [p for p_list in gathered_pred_lists for p in p_list]
        eval_metrics = {}
        if self.is_main_process:
            coco_gt = COCO(self.config.dataset_coco_gt_format_path)
            coco_pred = coco_gt.loadRes(predictions)
            coco_eval = COCOeval(coco_gt, coco_pred, iouType='segm')
            coco_eval.params.useCats = 0  # ignore categories as they are not predicted in ref-vos task
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()
            ap_labels = ['mAP 0.5:0.95', 'AP 0.5', 'AP 0.75', 'AP 0.5:0.95 S', 'AP 0.5:0.95 M', 'AP 0.5:0.95 L']
            ap_metrics = coco_eval.stats[:6]
            eval_metrics = {l: m for l, m in zip(ap_labels, ap_metrics)}
            if self.config.calculate_precision_and_iou_metrics:
                precision_at_k, overall_iou, mean_iou = calculate_precision_at_k_and_iou_metrics(coco_gt, coco_pred)
                eval_metrics.update({f'P@{k}': m for k, m in zip([0.5, 0.6, 0.7, 0.8, 0.9], precision_at_k)})
                eval_metrics.update({'overall_iou': overall_iou, 'mean_iou': mean_iou})
            print(eval_metrics)
        if self.distributed:
            dist.barrier()  # sync all processes before starting a new epoch or exiting
        return eval_metrics
        
    @torch.no_grad()
    def evaluate_refer_youtube_vos(self):
        self.model.eval()
        predictions = []
        for batch_dict in tqdm(self.data_loader_val, disable=not self.is_main_process):
            samples = batch_dict['samples'].to(self.device)
            # valid_indices = torch.arange(len(samples.tensors)).to(self.device)
            targets = to_device(batch_dict['targets'], self.device)
            valid_indices = None
            text_queries = batch_dict['text_queries']
            outputs = self.model(samples, valid_indices, text_queries, targets)
            videos_metadata = batch_dict['videos_metadata']
            sample_shape_with_padding = samples.tensors.shape[-2:]
            preds_by_video = self.postprocessor(outputs, videos_metadata, sample_shape_with_padding)
            predictions.extend(preds_by_video)
        # next, save the predictions
        validation_output_dir = path.join(self.output_dir_path, 'validation_outputs')
        epoch_validation_output_dir = path.join(validation_output_dir, f'epoch_{self.epoch}')
        annotations_dir = path.join(epoch_validation_output_dir, 'Annotations')
        print('saving predictions...')
        for p in tqdm(predictions, disable=not self.is_main_process):
            pred_dir_path = path.join(annotations_dir, p['video_id'], p['exp_id'])
            os.makedirs(pred_dir_path, exist_ok=True)
            for f_mask, f_idx in zip(p['pred_masks'], p['frame_indices']):
                pred_mask_path = path.join(pred_dir_path, f'{f_idx}.png')
                pred_mask = Image.fromarray((255 * f_mask.squeeze()).numpy())
                pred_mask.save(pred_mask_path)
        if self.distributed:
            dist.barrier()  # make sure all processes finished saving their predictions before creating the zip file
        if self.is_main_process:
            print('creating a zip file with the predictions...')
            # create zip file to be submitted to refer-youtube-vos validation server:
            zip_file_path = path.join(validation_output_dir, f'submission_epoch_{self.epoch}')
            shutil.make_archive(zip_file_path, 'zip', root_dir=epoch_validation_output_dir, base_dir='Annotations')
            print('a zip file was successfully created.')
            shutil.rmtree(epoch_validation_output_dir)  # remove the uncompressed annotations for memory efficiency
        if self.distributed:
            dist.barrier()  # sync all processes before starting a new epoch or exiting
        #self.model.module.backbone.running_mode = self.config.running_mode
        return {}  # return an empty metrics dict as all validation metrics will be computed on the server later

    @torch.no_grad()
    def evaluate_refer_youtube_vos_ema(self):
        self.model_ema.eval()
        predictions = []
        for batch_dict in tqdm(self.data_loader_val, disable=not self.is_main_process):
            samples = batch_dict['samples'].to(self.device)
            # valid_indices = torch.arange(len(samples.tensors)).to(self.device)
            targets = to_device(batch_dict['targets'], self.device)
            valid_indices = None
            text_queries = batch_dict['text_queries']
            outputs = self.model_ema(samples, valid_indices, text_queries, targets)
            videos_metadata = batch_dict['videos_metadata']
            sample_shape_with_padding = samples.tensors.shape[-2:]
            preds_by_video = self.postprocessor(outputs, videos_metadata, sample_shape_with_padding)
            predictions.extend(preds_by_video)
        # next, save the predictions
        validation_output_dir = path.join(self.output_dir_path, 'validation_ema_outputs')
        epoch_validation_output_dir = path.join(validation_output_dir, f'epoch_{self.epoch}')
        annotations_dir = path.join(epoch_validation_output_dir, 'Annotations')
        print('saving predictions...')
        for p in tqdm(predictions, disable=not self.is_main_process):
            pred_dir_path = path.join(annotations_dir, p['video_id'], p['exp_id'])
            os.makedirs(pred_dir_path, exist_ok=True)
            for f_mask, f_idx in zip(p['pred_masks'], p['frame_indices']):
                pred_mask_path = path.join(pred_dir_path, f'{f_idx}.png')
                pred_mask = Image.fromarray((255 * f_mask.squeeze()).numpy())
                pred_mask.save(pred_mask_path)
        if self.distributed:
            dist.barrier()  # make sure all processes finished saving their predictions before creating the zip file
        if self.is_main_process:
            print('creating a zip file with the predictions...')
            # create zip file to be submitted to refer-youtube-vos validation server:
            zip_file_path = path.join(validation_output_dir, f'submission_epoch_{self.epoch}')
            shutil.make_archive(zip_file_path, 'zip', root_dir=epoch_validation_output_dir, base_dir='Annotations')
            print('a zip file was successfully created.')
            shutil.rmtree(epoch_validation_output_dir)  # remove the uncompressed annotations for memory efficiency
        if self.distributed:
            dist.barrier()  # sync all processes before starting a new epoch or exiting
        #self.model.module.backbone.running_mode = self.config.running_mode
        return {}  # return

    def to_device(self, sample):
        if isinstance(sample, torch.Tensor):
            sample = sample.to(self.device)
        elif isinstance(sample, tuple) or isinstance(sample, list):
            sample = [self.to_device(s) for s in sample]
        return sample

    def load_checkpoint(self, checkpoint_path, total_epoch=None):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.epoch = checkpoint['epoch'] + 1  # the epoch after the one saved is about to begin
        if total_epoch == None:
            self.total_epochs = checkpoint['total_epochs']
        else:
            self.total_epochs = total_epoch
        if self.dataset_name == 'a2d_sentences':
            self.best_mAP = checkpoint['best_mAP']
        else:  # refer-youtube-vos
            self.best_loss = checkpoint['best_loss']
        model_without_ddp = self.model.module if isinstance(self.model, DDP) else self.model
        model_without_ddp.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.grad_scaler.load_state_dict(checkpoint['grad_scaler_state_dict'])

    def save_checkpoint(self, epoch_score):
        if not self.is_main_process:
            return
        is_best = False
        model_without_ddp = self.model.module if isinstance(self.model, DDP) else self.model
        checkpoint_dict = {
            'epoch': self.epoch,
            'total_epochs': self.total_epochs,
            'model_state_dict': model_without_ddp.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'grad_scaler_state_dict': self.grad_scaler.state_dict()
        }
        if self.model_ema:
            model_ema_without_ddp = self.model_ema.module if isinstance(self.model_ema, DDP) else self.model_ema
            checkpoint_dict["ema_model_state_dict"] = model_ema_without_ddp.state_dict()
        if self.dataset_name == 'a2d_sentences':
            is_best_mAP = epoch_score > self.best_mAP
            if is_best_mAP:
                self.best_mAP = epoch_score
                is_best = True
            checkpoint_dict['best_mAP'] = self.best_mAP
        else:  # refer-youtube-vos
            is_best_loss = epoch_score < self.best_loss
            if is_best_loss:
                self.best_loss = epoch_score
                is_best = True
            checkpoint_dict['best_loss'] = self.best_loss
        filename = self.get_checkpoint_filename()
        torch.save(checkpoint_dict, filename)
        print(f'saved checkpoint: {filename}')
        if is_best:
            best_filename = self.get_checkpoint_filename(is_best=True)
            shutil.copyfile(filename, best_filename)
        self.remove_extra_checkpoints()

    def get_checkpoint_filename(self, is_best=False):
        basename = 'best' if is_best else f'{self.epoch:02d}'
        return os.path.join(self.checkpoint_dir_path, f'{basename}.pth.tar')

    def remove_extra_checkpoints(self):
        filenames = sorted(os.listdir(self.checkpoint_dir_path))
        max_num_checkpoints = 5
        num_files_to_remove = max(0, len(filenames) - max_num_checkpoints)
        for filename in filenames[:num_files_to_remove]:
            os.remove(os.path.join(self.checkpoint_dir_path, filename))

    def clear_memory(self):
        if self.backbone_name == 'video-swin-t' or self.backbone_name == 'video-swin-s' or self.backbone_name == 'video-swin-b':
            compute_mask.cache_clear()  # empty cache of SwinT
        gc.collect()
        torch.cuda.empty_cache()

def pre_trained_model_to_finetune(checkpoint, args):
    checkpoint = checkpoint['model_state_dict']
    # only delete the class_embed since the finetuned dataset has different num_classes
    num_layers = args.DeformTransformer['dec_layers'] + 1 if args.DeformTransformer['two_stage'] else args.DeformTransformer['dec_layers']
    for l in range(num_layers):
        del checkpoint["class_embed.{}.weight".format(l)]
        del checkpoint["class_embed.{}.bias".format(l)]

    return checkpoint

def init_process_group_and_set_device(world_size, process_id, device_id, config):
    """
    This function needs to be called on each spawned process to initiate learning using DistributedDataParallel.
    The function initiates the process' process group and assigns it a single GPU to use during training.
    """
    config.world_size = world_size
    config.rank = process_id
    torch.cuda.set_device(device_id)
    device = torch.device(f'cuda:{device_id}')
    config.device = device
    if world_size > 1:
        config.distributed = True
        torch.distributed.init_process_group(
            torch.distributed.Backend.NCCL,
            world_size=world_size,
            rank=process_id
        )
        torch.distributed.barrier(device_ids=[device_id])
        utils.setup_for_distributed(config.rank == 0)
    else:
        config.distributed = False
    return device

def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out

def to_device(sample, device):
    if isinstance(sample, torch.Tensor):
        sample = sample.to(device)
    elif isinstance(sample, tuple) or isinstance(sample, list):
        sample = [to_device(s, device) for s in sample]
    elif isinstance(sample, dict):
        sample = {k: to_device(v, device) for k, v in sample.items()}
    return sample