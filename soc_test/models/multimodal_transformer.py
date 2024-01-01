"""
MTTR Multimodal Transformer class.
Modified from DETR https://github.com/facebookresearch/detr
"""
import copy
import os
from typing import Optional
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from einops import rearrange, repeat

from models.position_encoding import PositionEmbeddingSine2D

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # this disables a huggingface tokenizer warning (printed every epoch)


class MultimodalTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.d_model = config['d_model']
        self.use_multi = config['use_multi']

        encoder_layer = TransformerEncoderLayer(d_model = config['d_model'], nheads= config['nheads'])
        self.encoder = TransformerEncoder(encoder_layer, config['num_encoder_layers'])
        decoder_layer = TransformerDecoderLayer(d_model = config['d_model'], nheads = config['nheads'])
        self.decoder = TransformerDecoder(decoder_layer, config['num_decoder_layers'], norm=nn.LayerNorm(self.d_model),
                                          return_intermediate=True)
        self.pos_encoder_2d = PositionEmbeddingSine2D()
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, vid_embeds, vid_pad_mask, txt_memory, txt_pad_mask, text_sentence_feature, obj_queries):
        device = vid_embeds.device
        t, b, _, h, w = vid_embeds.shape
        if self.use_multi:
            # add temporal dim to txt memory & padding mask:
            txt_memory = repeat(txt_memory, 's b c -> s (t b) c', t=t)
            txt_pad_mask = repeat(txt_pad_mask, 'b s -> (t b) s', t=t)

            vid_embeds = rearrange(vid_embeds, 't b c h w -> (h w) (t b) c')
            # Concat the image & text embeddings on the sequence dimension
            encoder_src_seq = torch.cat((vid_embeds, txt_memory), dim=0)
            seq_mask = torch.cat((rearrange(vid_pad_mask, 't b h w -> (t b) (h w)'), txt_pad_mask), dim=1)
            # vid_pos_embed is: [T*B, H, W, d_model]
            vid_pos_embed = self.pos_encoder_2d(rearrange(vid_pad_mask, 't b h w -> (t b) h w'), self.d_model)
            # use zeros in place of pos embeds for the text sequence:
            pos_embed = torch.cat((rearrange(vid_pos_embed, 't_b h w c -> (h w) t_b c'), torch.zeros_like(txt_memory)), dim=0)

            memory = self.encoder(encoder_src_seq, src_key_padding_mask=seq_mask, pos=pos_embed)  # [S, T*B, C] S=H*W+L
            vid_memory = rearrange(memory[:h*w, :, :], '(h w) (t b) c -> t b c h w', h=h, w=w, t=t, b=b)
            txt_memory = memory[h*w:, :, :] 
        else:
            vid_embeds = rearrange(vid_embeds, 't b c h w -> (h w) (t b) c')
            encoder_src_seq = vid_embeds
            seq_mask = rearrange(vid_pad_mask, 't b h w -> (t b) (h w)')
            vid_pos_embed = self.pos_encoder_2d(rearrange(vid_pad_mask, 't b h w -> (t b) h w'), self.d_model)
            pos_embed = rearrange(vid_pos_embed, 't_b h w c -> (h w) t_b c')
            memory = self.encoder(encoder_src_seq, src_key_padding_mask=seq_mask, pos=pos_embed)  # [H*W, T*B, C]
            vid_memory = rearrange(memory, '(h w) (t b) c -> t b c h w', h=h, w=w, t=t, b=b)
        txt_memory = rearrange(txt_memory, 's t_b c -> t_b s c')
        txt_memory = [t_mem[~pad_mask] for t_mem, pad_mask in zip(txt_memory, txt_pad_mask)]
        #[(L1 C) (L1 C) T (L2 C) T] -> (S L C) 
        # add T*B dims to query embeds (was: [N, C], where N is the number of object queries):
        obj_queries = repeat(obj_queries, 'n c -> n (t b) c', t=t, b=b)
        num_queries = obj_queries.size()[0]
        tgt = torch.zeros_like(obj_queries)  # [N, T*B, C]
        # tgt = repeat(text_sentence_feature, 'b c -> n (t b) c', t=t, n=num_queries)

        # hs is [L, N, T*B, C] where L is number of layers in the decoder
        hs = self.decoder(tgt, memory, memory_key_padding_mask=seq_mask, pos=pos_embed, query_pos=obj_queries)
        hs = rearrange(hs, 'l n (t b) c -> l t b n c', t=t, b=b) #[layer, frame, batch, num_queries, channel]
        return hs, vid_memory, txt_memory, text_sentence_feature


    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nheads, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nheads, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nheads, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nheads, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nheads, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
