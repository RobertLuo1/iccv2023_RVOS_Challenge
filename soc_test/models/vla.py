import torch.nn as nn
from einops import rearrange
import torch
import torch.nn.functional as F
from typing import List, Optional
from torch import Tensor


class SpatialImageLanguageAttention(nn.Module):
    def __init__(self, v_in_channels, l_in_channels, key_channels, value_channels, out_channels=None, num_heads=1):
        super(SpatialImageLanguageAttention, self).__init__()
        # x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l, 1)
        self.v_in_channels = v_in_channels
        self.l_in_channels = l_in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.value_channels = value_channels
        self.num_heads = num_heads
        if out_channels is None:
            self.out_channels = self.value_channels

        # Keys: language features: (B, l_in_channels, #words)
        # avoid any form of spatial normalization because a sentence contains many padding 0s
        self.f_key = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.key_channels, kernel_size=1, stride=1),
        )

        # Queries: visual features: (B, H*W, v_in_channels)
        self.f_query = nn.Sequential(
            nn.Conv1d(self.v_in_channels, self.key_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.key_channels),
        )

        # Values: language features: (B, l_in_channels, #words)
        self.f_value = nn.Sequential(
            nn.Conv1d(self.l_in_channels, self.value_channels, kernel_size=1, stride=1),
        )

        # Out projection
        self.W = nn.Sequential(
            nn.Conv1d(self.value_channels, self.out_channels, kernel_size=1, stride=1),
            nn.InstanceNorm1d(self.out_channels),
        )

    def forward(self, x, l, l_mask):
        # x shape: (B, H*W, v_in_channels)
        # l input shape: (B, l_in_channels, N_l)
        # l_mask shape: (B, N_l)
        B, HW = x.size(0), x.size(1)
        x = x.permute(0, 2, 1)  # (B, key_channels, H*W)
        l_mask = l_mask.unsqueeze(-1) #(B N_l 1)
        l_mask = l_mask.permute(0, 2, 1)  # (B, N_l, 1) -> (B, 1, N_l)

        query = self.f_query(x)  # (B, key_channels, H*W) if Conv1D
        query = query.permute(0, 2, 1)  # (B, H*W, key_channels)
        key = self.f_key(l)  # (B, key_channels, N_l)
        value = self.f_value(l)  # (B, self.value_channels, N_l)
        key = key * l_mask  # (B, key_channels, N_l)
        value = value * l_mask  # (B, self.value_channels, N_l)
        n_l = value.size(-1)
        query = query.reshape(B, HW, self.num_heads, self.key_channels//self.num_heads).permute(0, 2, 1, 3)
        # (b, num_heads, H*W, self.key_channels//self.num_heads)
        key = key.reshape(B, self.num_heads, self.key_channels//self.num_heads, n_l)
        # (b, num_heads, self.key_channels//self.num_heads, n_l)
        value = value.reshape(B, self.num_heads, self.value_channels//self.num_heads, n_l)
        # # (b, num_heads, self.value_channels//self.num_heads, n_l)
        l_mask = l_mask.unsqueeze(1)  # (b, 1, 1, n_l)

        sim_map = torch.matmul(query, key)  # (B, self.num_heads, H*W, N_l)
        sim_map = (self.key_channels ** -.5) * sim_map  # scaled dot product

        sim_map = sim_map + (1e4*l_mask - 1e4)  # assign a very small number to padding positions
        sim_map = F.softmax(sim_map, dim=-1)  # (B, num_heads, h*w, N_l)
        out = torch.matmul(sim_map, value.permute(0, 1, 3, 2))  # (B, num_heads, H*W, self.value_channels//num_heads)
        out = out.permute(0, 2, 1, 3).contiguous().reshape(B, HW, self.value_channels)  # (B, H*W, value_channels)
        out = out.permute(0, 2, 1)  # (B, value_channels, HW)
        out = self.W(out)  # (B, value_channels, HW)
        out = out.permute(0, 2, 1)  # (B, HW, value_channels)

        return out


class VisualLanguageALignment(nn.Module):
    def __init__(self, channel, language_emb, nhead=8, dropout=0.0) -> None:
        super().__init__()

        self.visual_proj = nn.Sequential(
            nn.Conv1d(channel, channel, 1, 1),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
        #video encoder [96 192 384] language 768
        # self.language_resizer = nn.Sequential(
        #     nn.Linear(language_emb, channel),
        #     nn.ReLU(inplace=True) #768
        # )
        
        self.vl_cross_attn = SpatialImageLanguageAttention(
            v_in_channels = channel, l_in_channels = language_emb,
            key_channels = channel, value_channels = channel, out_channels = channel,
            num_heads = 1
        )
        
        self.lv_cross_attn = SpatialImageLanguageAttention(
             v_in_channels = channel, l_in_channels = language_emb,
            key_channels = channel, value_channels = channel, out_channels = channel,
            num_heads = 1
        )
        # self.dropouts = nn.ModuleList()
        # self.norms = nn.ModuleList()
        # for _ in range(num_frames):
        #     self.dropouts.append(nn.Dropout(dropout))
        #     self.norms.append(nn.LayerNorm(channel))
        # self.recover = nn.Sequential(
        #     nn.Linear(channel, channel),
        #     nn.ReLU(inplace=True),
        # )

        # self.projector = nn.Sequential(
        #     nn.Linear(channel, channel),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(channel, channel),
        #     nn.Tanh()
        # )

        # self.norm1 = nn.LayerNorm(channel)

    def forward(self, vid_embed, visual_masks, language_embed, language_mask):
        """
        vid_embed: [(t h w) b c]
        language_embed: [l b c]
        language_mask: [b l]
        visual_masks: [l b]
        """
        # vid_embed = rearrange(vid_embed, 'b c t h w -> t (h w) b c')
        visual_masks = rearrange(visual_masks, 'l b -> b l')
        vid_embed = rearrange(vid_embed, 'l b c -> b c l')
        vid_embed = self.visual_proj(vid_embed)
        vid_embed = rearrange(vid_embed, "b c l -> b l c")
        language_embed = rearrange(language_embed, 'l b c -> b c l')
        
        visual_tgt = self.vl_cross_attn(
            x = vid_embed,
            l = language_embed,
            l_mask = language_mask,
        ) #[b thw c]

        visual_tgt = visual_tgt * vid_embed
        
        ####--------------------------------------
        vid_embed = rearrange(vid_embed, 'b l c -> b c l')
        language_embed = rearrange(language_embed, 'b c l -> b l c')
        language_tgt = self.lv_cross_attn(
            x = language_embed,
            l = vid_embed,
            l_mask = visual_masks
        )

        language_tgt =  language_tgt * language_embed #[B L C]
            # visual_tgt = visual_tgt * vid_embed[idx]
            # if running_mode == 'train' or running_mode == 'resume_train':
            #     visual_tgt = self.dropouts[idx](vid_embed[idx]) + visual_tgt
            #     visual_tgt = self.norms[idx](visual_tgt)
            # else:
            #     visual_tgt = self.dropouts[0](vid_embed[idx]) + visual_tgt
            #     visual_tgt = self.norms[0](visual_tgt)
            # visual_tgts.append(visual_tgt)

        # visual_tgts = torch.stack(visual_tgts,dim=0) #[t hw b c]
        # visual_tgt = self.recover(visual_tgt)
        # visual_tgts = self.projector(visual_tgts) * visual_tgts

        return visual_tgt, language_tgt

class VisionLanguageFusionModule(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=None,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt * tgt2
        return tgt

class LanguageVisionModule(nn.Module):
    """
    fuse the langauge and visual input
    """
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=0.1)
    
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos
    
    def forward(self, tgt, memory,
                memory_key_padding_mask: Optional[Tensor] = None,
                tgt_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        
        tgt2 = self.cross_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=None,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt * tgt2

        tgt3 = self.self_attn(
            query = tgt,
            key = tgt,
            value = tgt,
            key_padding_mask = tgt_padding_mask
        )[0]


        tgt = tgt + tgt3
        return tgt