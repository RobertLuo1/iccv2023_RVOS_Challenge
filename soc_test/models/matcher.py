"""
Modified from DETR https://github.com/facebookresearch/detr
Module to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
from misc import nested_tensor_from_tensor_list, interpolate, box_cxcywh_to_xyxy, generalized_box_iou
from einops import rearrange


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_is_referred: float = 1, cost_dice: float = 1, 
                 cost_assist: float = 1, cost_box: float = 1, cost_giou: float = 1, num_classes: int = 1):
        """Creates the matcher

        Params:
            cost_is_referred: This is the relative weight of the reference cost in the total matching cost
            cost_dice: This is the relative weight of the dice cost in the total matching cost
        """
        super().__init__()
        self.num_classes = num_classes
        self.cost_is_referred = cost_is_referred
        self.cost_dice = cost_dice
        self.cost_assist = cost_assist
        self.cost_box = cost_box
        self.cost_giou = cost_giou

        assert cost_is_referred != 0 or cost_dice != 0, "all costs cant be 0"

    @torch.inference_mode()
    def forward(self, outputs, targets, text_refer):
        """ Performs the matching

        Params:
            outputs: A dict that contains at least these entries:
                 "pred_is_referred": Tensor of dim [time, batch_size, num_queries, 2] with the reference logits
                 "pred_masks": Tensor of dim [time, batch_size, num_queries, H, W] with the predicted masks logits

            targets: A list of lists of targets (outer - time steps, inner - batch samples). each target is a dict
                     which contain mask and reference ground truth information for a single frame.

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_masks)
        """
        t, bs, num_queries = outputs["pred_masks"].shape[:3] #T B N H_pred, W_pred

        # We flatten to compute the cost matrices in a batch
        out_masks = outputs["pred_masks"].flatten(1, 2)  # [t, batch_size * num_queries, mask_h, mask_w]

        # preprocess and concat the target masks
        #t_step_batch: ({},{},{}) v: ({}
        tgt_masks = [[m for v in t_step_batch for m in v["masks"].unsqueeze(1)] for t_step_batch in targets]
        # pad the target masks to a uniform shape
        tgt_masks, valid = list(zip(*[nested_tensor_from_tensor_list(t).decompose() for t in tgt_masks]))
        tgt_masks = torch.stack(tgt_masks).squeeze(2) #[t, instance_numbers, h, w]
        
        # upsample predicted masks to target mask size
        out_masks = interpolate(out_masks, size=tgt_masks.shape[-2:], mode="bilinear", align_corners=False)

        tgt_boxes = [torch.stack([m for v in t_step_batch for m in v["boxes"]]) for t_step_batch in targets]
        tgt_boxes = torch.stack(tgt_boxes) #[T, instances(all objetc in batch), 4]

        output_boxes = outputs['pred_boxes']
        # Compute the soft-tokens cost:
        if self.cost_is_referred > 0:
            cost_is_referred = text_refer_cost(outputs, targets)
        else:
            cost_is_referred = 0
        if self.cost_assist > 0:
            cost_assist = compute_label_cost(outputs, targets, self.num_classes)
        else:
            cost_assist = 0
        if self.cost_box > 0:
            cost_box = costs_box(output_boxes, tgt_boxes)
        else:
            cost_box = 0
        if self.cost_giou > 0:
            cost_giou = giou_cost(output_boxes, tgt_boxes)
        else:
            cost_giou = 0

        # Compute the DICE coefficient between the masks:
        if self.cost_dice > 0:
            cost_dice = -dice_coef(out_masks, tgt_masks)
        else:
            cost_dice = 0

        # Final cost matrix
        # C = self.cost_is_referred * cost_is_referred + self.cost_dice * cost_dice
        C = self.cost_is_referred * cost_is_referred + self.cost_assist * cost_assist + self.cost_dice * cost_dice + self.cost_box * cost_box + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu() #[b, query, i_total]

        num_traj_per_batch = [len(v["masks"]) for v in targets[0]]  # number of instance trajectories in each batch
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(num_traj_per_batch, -1))]
        device = out_masks.device
        return [(torch.as_tensor(i, dtype=torch.int64, device=device),
                 torch.as_tensor(j, dtype=torch.int64, device=device)) for i, j in indices]


def build_matcher(args):
    return HungarianMatcher(cost_is_referred=args.set_cost_is_referred, cost_dice=args.set_cost_dice, 
                            cost_assist=args.set_cost_is_referred_assist, cost_box = args.set_costs_box, cost_giou = args.set_costs_giou, num_classes=args.num_classes)


def dice_coef(inputs, targets, smooth=1.0):
    """
    Compute the DICE coefficient, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid().flatten(2).unsqueeze(2) #[t, batch_size * num_queries, 1, mask_h * mask_w]
    targets = targets.flatten(2).unsqueeze(1) #[t, 1, instacesa_num, h*w]
    numerator = 2 * (inputs * targets).sum(-1) #[t, bq, i]
    denominator = inputs.sum(-1) + targets.sum(-1)
    coef = (numerator + smooth) / (denominator + smooth)
    coef = coef.mean(0)  # average on the temporal dim to get instance trajectory scores
    return coef #[bq, i]

# def compute_vita_is_referred_cost(outputs, targets):
#     t = len(targets)
#     Bs, num_queries, _  = outputs['pred_is_referred'].size()
#     pred_is_referred = outputs['pred_is_referred'].softmax(dim=-1) #[B NQ 2]
#     loss_query_objects = []
#     for i in range(Bs):
#         refer_queries = pred_is_referred[i]
#         bs_target = [t_step[i] for t_step in targets]
        
            
    # pred_is_referred = rearrange(pred_is_referred, 'b nq c -> (b nq) c')
    # device = pred_is_referred.device
    # num_traj_per_batch = torch.tensor([len(v["masks"]) for v in targets[0]], device=device)
    # ref_indices = torch.tensor([v['referred_instance_idx'] for v in targets[0]], device=device)
    # is_ref_inst_visible = torch.stack([torch.stack([t['is_ref_inst_visible'] for t in t_step]) for t_step in targets]).permute(1, 0)
    # target_is_referred = torch.zeros((t, num_traj_per_batch, 2), device=device)
    # #all object is not referred
    # target_is_referred[:, :, :] = torch.tensor([0.0, 1.0], device=device)
    # for ref_idx, is_visible in zip(ref_indices, is_ref_inst_visible):
    #         is_visible = is_visible.nonzero().squeeze()
    #         target_is_referred[is_visible, ref_idx, :] = torch.tensor([1.0, 0.0], device=device) #[T num_objects 2]
    # cost_is_referred = -(pred_is_referred.unsqueeze(2) * target_is_referred.unsqueeze(1)).sum(dim=-1).mean(dim=0) #[b*nq numobject]

def text_refer_cost(outputs, targets):
    output_logit = outputs['pred_logit'] #[B Nq C]  
    text_feature = outputs['text_sentence_feature']
    device = text_feature.device
          #[b C] 
    bs = output_logit.shape[0]
    text_sentence_feature = text_feature.unsqueeze(1) #[B 1 C]

    query_text_sim = output_logit @ text_sentence_feature.transpose(1, 2) #[B Nq 1]
    alpha = 0.25
    gamma = 2.0
    # query_text_sim = query_text_sim.sigmoid()
    # query_text_sim = query_text_sim.flatten(0, 1)
    # neg_cost_class = (1 - alpha) * (query_text_sim ** gamma) * (-(1 - query_text_sim + 1e-8).log())
    # pos_cost_class = alpha * ((1 - query_text_sim) ** gamma) * (-(query_text_sim + 1e-8).log())
    # cost_text = pos_cost_class[:, [0]] - neg_cost_class[:, [0]]
    cost = query_text_sim.squeeze(-1).softmax(dim=-1).unsqueeze(2).flatten(0,1) #[B*Nq,1]

    #[bnq, 1]
    return -cost

def compute_label_cost(outputs, targets, num_classes):
    alpha = 0.25
    gamma = 2.0
    pred_label = outputs['pred_is_referred'].sigmoid() #[t, b, nq, K]
    device = pred_label.device
    t, b, nq, k = pred_label.shape
    pred_label = pred_label.flatten(1,2)#[t bnq k]
    neg_cost_class = (1 - alpha) * (pred_label ** gamma) * (-(1 - pred_label + 1e-8).log())
    pos_cost_class = alpha * ((1 - pred_label) ** gamma) * (-(pred_label + 1e-8).log())
    # pred_label = pred_label.flatten(1,2)#[t bnq k]
    if num_classes == 1: #a2d object is visible for all object
        if t == 1: #for a2d and coco pretrain
            cost_class_splits = pos_cost_class[: ,:, [0]] - neg_cost_class[:, :, [0]]
            cost_class_splits = torch.mean(cost_class_splits, dim=0) #average the t
            # cost_class_splits = cost_class_splits[0, ...]
            return cost_class_splits
        else: #for ref-youtube and joint frames is higher than 1
            # pred_label = pred_label.flatten(1,2)#[t bnq k]
            cost_class_splits = []
            is_ref_inst_visible = torch.stack([torch.stack([t['is_ref_inst_visible'] for t in t_step]) for t_step in targets]).permute(1, 0) #[instances, t]
            for idx, is_visible in enumerate(is_ref_inst_visible): #iterate batch
                is_visible = is_visible.bool()
                cost_class_split = pos_cost_class[is_visible, :, [0]] - neg_cost_class[is_visible, :, [0]]
                cost_class_splits.append(cost_class_split) # back to the one instance
            cost_class_splits = torch.stack(cost_class_splits, dim=-1) #stack in the instances size  
            cost_class_splits = torch.mean(cost_class_splits, dim=0) #average the t
            return cost_class_splits
    else: #refyoutube and coco
        # target_select = torch.zeros(size=(t, b*nq, num_classes)).to(device)
        cost_class_splits = []
        targets_label = [torch.stack([m for v in t_step_batch for m in v["labels"]]) for t_step_batch in targets] #[({}, {})] #[instances]
        targets_label = torch.stack(targets_label, dim=0).permute(1, 0) #[t, instances] -> [instance, t]
        is_ref_inst_visible = torch.stack([torch.stack([t['is_ref_inst_visible'] for t in t_step]) for t_step in targets]).permute(1, 0) #[instances, t]
        # pos_neg_div_class = pos_cost_class - neg_cost_class
        for idx, (tgt_label, is_visible) in enumerate(zip(targets_label, is_ref_inst_visible)):
            # is_visible = is_visible.nonzero().squeeze(1)
            # target_select[is_visible, :, tgt_label] = torch.tensor([1.0], device=device)
            cost_class_split = pos_cost_class[is_visible, :, tgt_label[0]] - neg_cost_class[is_visible, :, tgt_label[0]] #[t bq instances]
            # cost_class_split = (pos_neg_div_class * target_select)[:, :, tgt_label]
            cost_class_splits.append(cost_class_split)
        cost_class_splits = torch.stack(cost_class_splits, dim=-1) #[t bnq instance]

        cost_class_splits = torch.mean(cost_class_splits,dim=0) #average the mean

        return cost_class_splits

def costs_box(inputs, targets):
    """
    Compute the L1_cost and giou cost of the box
    @input: [T B N 4]
    @targets: [T instance 4]
    """
    inputs = inputs.flatten(1, 2)
    cost_bbox = torch.cdist(inputs, targets, p=1)
    cost_bbox = cost_bbox.mean(0)

    return cost_bbox 

def giou_cost(inputs, targets):
    """
    Compute the GIOU cost of box
    @input: [T B*N 4]
    @targets: [T instances(B instances) 4]
    """
    inputs = inputs.flatten(1, 2)
    T, bn, c = inputs.shape
    frame_costs = []
    for t in range(T):
        input = inputs[t]
        target = targets[t]
        input = box_cxcywh_to_xyxy(input)
        target = box_cxcywh_to_xyxy(target)
        cost_giou = -generalized_box_iou(input, target)
        frame_costs.append(cost_giou)

    costs = torch.stack(frame_costs).mean(0)
    return costs 

    

def compute_is_referred_cost(outputs, targets):
    pred_is_referred = outputs['pred_is_referred'].flatten(1, 2).softmax(dim=-1)  # [t,bnq, 2] 
    device = pred_is_referred.device
    t = pred_is_referred.shape[0]
    # number of instance trajectories in each batch
    num_traj_per_batch = torch.tensor([len(v["masks"]) for v in targets[0]], device=device) #for refer-Youtube all onjects included in a masks
    total_trajectories = num_traj_per_batch.sum()
    # note that ref_indices are shared across time steps:
    ref_indices = torch.tensor([v['referred_instance_idx'] for v in targets[0]], device=device)
    # convert ref_indices to fit flattened batch targets:
    ref_indices += torch.cat((torch.zeros(1, dtype=torch.long, device=device), num_traj_per_batch.cumsum(0)[:-1]))
    # number of instance trajectories in each batch
    target_is_referred = torch.zeros((t, total_trajectories, 2), device=device)
    # 'no object' class by default (for un-referred objects)
    target_is_referred[:, :, :] = torch.tensor([0.0, 1.0], device=device)
    if 'is_ref_inst_visible' in targets[0][0]:  # visibility labels are available per-frame for the referred object:
        is_ref_inst_visible = torch.stack([torch.stack([t['is_ref_inst_visible'] for t in t_step]) for t_step in targets]).permute(1, 0)
        for ref_idx, is_visible in zip(ref_indices, is_ref_inst_visible):
            is_visible = is_visible.nonzero().squeeze()
            target_is_referred[is_visible, ref_idx, :] = torch.tensor([1.0, 0.0], device=device)
    else:  # assume that the referred object is visible in every frame:
        target_is_referred[:, ref_indices, :] = torch.tensor([1.0, 0.0], device=device)
    cost_is_referred = -(pred_is_referred.unsqueeze(2) * target_is_referred.unsqueeze(1)).sum(dim=-1).mean(dim=0)

                        # [t,bnq,1,2]  * [t,1,i,2]  [t,bnq,i,2] [bnq, i]
    return cost_is_referred
