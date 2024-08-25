import numpy as np
import torch
from torch import nn

from ..builder import SUBMODULES
from ..builder import build_attention

from .diffusion_transformer import DiffusionTransformer
from .diffusion_transformer import FFN


class DecoderLayer(nn.Module):

    def __init__(self, sa_block_cfg=None, ca_block_cfg=None, ffn_cfg=None):
        super().__init__()
        self.sa_block = build_attention(sa_block_cfg)
        self.ca_block = build_attention(ca_block_cfg)
        if ffn_cfg is not None:
            self.ffn_channel = FFN(**ffn_cfg)
            self.ffn_temporal = FFN(**ffn_cfg)
        else:
            self.ffn_channel = None
            self.ffn_temporal = None

    def forward(self, **kwargs):
        if self.sa_block is not None:
            
            kwargs.update({'src_mask': torch.ones(kwargs['x'].shape[0], kwargs['x'].shape[-1], kwargs['x'].shape[-2]).to(kwargs['x'].device)})
            kwargs.update({'x': kwargs['x'].transpose(-1, -2)})
            x = self.sa_block(**kwargs)
            x = x.transpose(-1, -2)
            kwargs.update({'x': x})
        if self.ffn_channel is not None:
            x = self.ffn_channel(**kwargs)
        if self.ca_block is not None:
            
            x = self.ca_block(**kwargs)
            kwargs.update({'x': x})
        if self.ffn_temporal is not None:
            x = self.ffn_temporal(**kwargs)
        return x


@SUBMODULES.register_module()
class MCMTransformer(DiffusionTransformer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build_temporal_blocks(self, sa_block_cfg, ca_block_cfg, ffn_cfg):
        self.temporal_decoder_blocks = nn.ModuleList()
        for i in range(self.num_layers):
            self.temporal_decoder_blocks.append(
                DecoderLayer(sa_block_cfg=sa_block_cfg,
                             ca_block_cfg=ca_block_cfg,
                             ffn_cfg=ffn_cfg))

    def get_precompute_condition(self,
                                 text=None,
                                 xf_proj=None,
                                 xf_out=None,
                                 device=None,
                                 clip_feat=None,
                                 **kwargs):
        if xf_proj is None or xf_out is None:
            xf_proj, xf_out = self.encode_text(text, clip_feat, device)
        return {'xf_proj': xf_proj, 'xf_out': xf_out}

    def post_process(self, motion):
        if self.post_process_cfg is not None:
            if self.post_process_cfg.get("unnormalized_infer", False):
                mean = torch.from_numpy(
                    np.load(self.post_process_cfg['mean_path']))
                mean = mean.type_as(motion)
                std = torch.from_numpy(
                    np.load(self.post_process_cfg['std_path']))
                std = std.type_as(motion)
            motion = motion * std + mean
        return motion

    def forward_train(self,
                      h=None,
                      src_mask=None,
                      emb=None,
                      xf_out=None,
                      **kwargs):
        B, T = h.shape[0], h.shape[1]
        for module in self.temporal_decoder_blocks:
            h = module(x=h, xf=xf_out, emb=emb, src_mask=src_mask)
        output = self.out(h).view(B, T, -1).contiguous()
        return output

    def forward_test(self,
                     h=None,
                     src_mask=None,
                     emb=None,
                     xf_out=None,
                     **kwargs):
        B, T = h.shape[0], h.shape[1]
        for module in self.temporal_decoder_blocks:
            h = module(x=h, xf=xf_out, emb=emb, src_mask=src_mask)
        output = self.out(h).view(B, T, -1).contiguous()
        return output