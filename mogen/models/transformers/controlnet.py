import re
import torch
import torch.nn as nn

from copy import deepcopy
from torch import Tensor
from torch.nn import Module, Linear, init
from typing import Any, Mapping


from mogen.models.utils.misc import set_requires_grad, zero_module
from mogen.models.utils.position_encoding import timestep_embedding
from .stmogen import DecoderLayer, STMoGenTransformer

from mogen.models.utils.blocks import WavEncoder
from ..utils.blocks import PatchEmbed1D

def fill_tensor_with_mask_efficient(h, c, mask):
    
    hidden_dim = c.size(2)
    assert mask.sum(dim=1).eq(c.size(1)).all()
    _, c_indices = torch.where(mask == 1)
    c_new = torch.zeros_like(h)
    c_new[mask.bool()] = c.contiguous().view(-1, hidden_dim)[c_indices]
    
    return c_new, c_indices



class ControlT2MBlock(Module):
    def __init__(self, base_block: DecoderLayer, block_index: 0, latent_dim: 512, cfg: None) -> None:
        super().__init__()
        

        self.copied_block = DecoderLayer(ca_block_cfg=cfg.model.model['ca_block_cfg'],
                            ffn_cfg=cfg.model.model.ffn_cfg)
        base_block_state_dict = base_block.state_dict()
        self.copied_block.load_state_dict(base_block_state_dict)
        self.block_index = block_index

        set_requires_grad(self.copied_block, True)
        self.copied_block.train()
        
        self.hidden_size = latent_dim
        if self.block_index == 0:
            self.before_proj = Linear(self.hidden_size, self.hidden_size)
            init.zeros_(self.before_proj.weight)
            init.zeros_(self.before_proj.bias)
        self.after_proj = Linear(self.hidden_size, self.hidden_size) 
        init.zeros_(self.after_proj.weight)
        init.zeros_(self.after_proj.bias)

    def forward(self, 
                x,
                xf,
                emb,
                src_mask,
                cond_type,
                motion_length,
                num_intervals,
                c=None,
                **kwargs):
    
        if self.block_index == 0:
            
            c = self.before_proj(c)
            
            c = self.copied_block(x=x + c,
                                  xf=xf,
                                  emb=emb,
                                  src_mask=src_mask,
                                  cond_type=cond_type,
                                  motion_length=motion_length,
                                  num_intervals=num_intervals)
            c_skip = self.after_proj(c)
        else:
            
            
            c = self.copied_block(x=c,
                                  xf=xf,
                                  emb=emb,
                                  src_mask=src_mask,
                                  cond_type=cond_type,
                                  motion_length=motion_length,
                                  num_intervals=num_intervals)
            c_skip = self.after_proj(c)
        
        return c, c_skip
        
class ConditionEncoder(Module):
    def __init__(self, condition_encode_cfg) -> None:
        super().__init__()
        if condition_encode_cfg.dataset_name == 'beats2':
            if condition_encode_cfg.condition_pre_encode_type == 'wav':
                self.pre_encoder  = WavEncoder(out_dim=condition_encode_cfg.condition_latent_dim, audio_in=condition_encode_cfg.control_cond_feats)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

    def forward(
            self, 
            condition,
        ):
        return self.pre_encoder(condition)


class ControlT2MHalf(Module):
    
    def __init__(self, base_model: STMoGenTransformer, 
            copy_blocks_num: int = 2, 
            control_cond_feats: int = 438, 
            cfg: dict = {}, 
            joint_embed_unfreeze: bool = True,
            unfreeze_mode = "all",
        ) -> None:
        
        super().__init__()
        self.cfg = cfg
        self.base_model = base_model.eval()
        self.controlnet = []
        self.copy_blocks_num = copy_blocks_num
        self.total_blocks_num = len(self.base_model.temporal_decoder_blocks)
        
        set_requires_grad(self.base_model.clip, False)
        set_requires_grad(self.base_model.text_pre_proj, False)
        set_requires_grad(self.base_model.textTransEncoder, False)
        set_requires_grad(self.base_model.text_ln, False)
        set_requires_grad(self.base_model.temporal_decoder_blocks, False)
        set_requires_grad(self.base_model.time_embed, False)
        if joint_embed_unfreeze:
            set_requires_grad(self.base_model.joint_embed, True)
            set_requires_grad(self.base_model.out, True)
            if unfreeze_mode != "all":
                set_requires_grad(self.base_model.joint_embed, False)
                set_requires_grad(self.base_model.out, False)
                set_requires_grad(self.base_model.joint_embed, True, mode=unfreeze_mode)
                set_requires_grad(self.base_model.out, True, mode=unfreeze_mode)
                
                
                
                
        else:
            set_requires_grad(self.base_model.joint_embed, False)
            set_requires_grad(self.base_model.out, False)
        
        if self.base_model.use_pos_embedding:
            self.base_model.sequence_embedding.requires_grad = False
            
        

        
        for i in range(copy_blocks_num):
            self.controlnet.append(ControlT2MBlock(self.base_model.temporal_decoder_blocks[i], i, self.base_model.latent_dim, cfg))
        self.controlnet = nn.ModuleList(self.controlnet)
        set_requires_grad(self.controlnet, True)
        self.controlnet.train()

        
        if cfg.condition_encode_cfg.condition_pre_encode:
            self.condition_pre_encoder = ConditionEncoder(cfg.condition_encode_cfg)
            set_requires_grad(self.condition_pre_encoder, True)
            self.condition_pre_encoder.train()
            self.control_cond_input = nn.Linear(in_features=cfg.condition_encode_cfg.condition_latent_dim, out_features=self.base_model.latent_dim)
            init.zeros_(self.control_cond_input.weight)
            init.zeros_(self.control_cond_input.bias)
            set_requires_grad(self.control_cond_input, True)
            self.control_cond_input.train()
        else:
            self.condition_pre_encoder = lambda x: x
            self.control_cond_input = nn.Linear(in_features=control_cond_feats, out_features=self.base_model.latent_dim)
            init.zeros_(self.control_cond_input.weight)
            init.zeros_(self.control_cond_input.bias)
            set_requires_grad(self.control_cond_input, True)
            self.control_cond_input.train()
        self.patch_size = cfg.get("patch_size", 1)
        if self.patch_size > 1:
            self.condition_patch = PatchEmbed1D(
                patch_size=(self.patch_size,),
                in_chans=self.base_model.latent_dim, 
                embed_dim=self.base_model.latent_dim
                )
        else:
            self.condition_patch = lambda x: x

    def forward_c(self, c, h, mask=None):
        c = self.condition_pre_encoder(c)
        c = self.control_cond_input(c)
        c = self.condition_patch(c)
        
        seq_len_c, seq_len_h = c.shape[1], h.shape[1]
        
        padding_len = seq_len_h - seq_len_c
        
        c_new = torch.cat([c, torch.zeros(c.shape[0], padding_len, c.shape[2]).to(h.device)], dim=-2)

        pos_len = seq_len_c 
        c_new[:, :pos_len, :] = c_new[:, :pos_len, :] + self.base_model.sequence_embedding.unsqueeze(0)[:, :pos_len, :]
        return c_new

    def forward(self,
                motion,
                timesteps,
                motion_mask=None,
                motion_length=None,
                num_intervals=1,
                c=None,
                **kwargs):
        
        """
        Forward pass of STMoGenTransformer.
        motion: (B, T, D) tensor of motion inputs (motion or latent representations of motion)
        timesteps: (B,) tensor of diffusion timesteps
        """
        T = motion.shape[1] // self.patch_size
        conditions = self.base_model.get_precompute_condition(device=motion.device,
                                                   **kwargs)
        if len(motion_mask.shape) == 2:
            src_mask = motion_mask.clone().unsqueeze(-1)
        else:
            src_mask = motion_mask.clone()

        if self.base_model.time_embedding_type == 'sinusoidal':
            emb = self.base_model.time_embed(
                timestep_embedding(timesteps, self.base_model.latent_dim))
        else:
            emb = self.base_model.time_embed(self.base_model.time_tokens(timesteps))

        if self.base_model.use_text_proj:
            emb = emb + conditions['xf_proj']

        
        h = self.base_model.joint_embed(motion)
        
        if c is not None:
            c = c.to(self.dtype)
            c = self.forward_c(c, h)

        
        
        

        if self.base_model.use_pos_embedding:
            h = h + self.base_model.sequence_embedding.unsqueeze(0)[:, :T, :]

        if self.controlnet.training:
            output = self.forward_train(h=h,
                                        src_mask=src_mask,
                                        emb=emb,
                                        timesteps=timesteps,
                                        motion_length=motion_length,
                                        num_intervals=num_intervals,
                                        c=c,
                                        **conditions)
        else:
            output = self.forward_test(h=h,
                                       src_mask=src_mask,
                                       emb=emb,
                                       timesteps=timesteps,
                                       motion_length=motion_length,
                                       num_intervals=num_intervals,
                                       c=c,
                                       **conditions)
        if self.base_model.use_residual_connection:
            output = motion + output
        return output
    
    def forward_train(self,
                      h=None,
                      src_mask=None,
                      emb=None,
                      xf_out=None,
                      motion_length=None,
                      num_intervals=1,
                      c=None,
                      **kwargs):
        B, T = h.shape[0], h.shape[1]
        T = T * self.patch_size
        cond_type = torch.randint(0, 100, size=(B, 1, 1)).to(h.device)

        
        last_h = h
        h = self.base_model.temporal_decoder_blocks[0](x=h,
                                                       xf=xf_out,
                                                       emb=emb,
                                                       src_mask=src_mask,
                                                       cond_type=cond_type,
                                                       motion_length=motion_length,
                                                       num_intervals=num_intervals)

        if c is not None:
            

            
            if self.cfg.condition_encode_cfg.condition_cfg:
                
                c_cond_type = (cond_type % 10 > 0).float()
                c = c * c_cond_type

            for index in range(1, self.copy_blocks_num + 1):
                c, c_skip = self.controlnet[index - 1](x=h,
                                                    xf=xf_out,
                                                    emb=emb,
                                                    src_mask=src_mask,
                                                    cond_type=cond_type,
                                                    motion_length=motion_length,
                                                    num_intervals=num_intervals,
                                                    c=c)
                h = self.base_model.temporal_decoder_blocks[index](x=h + c_skip,
                                                                   xf=xf_out,
                                                                   emb=emb,
                                                                   src_mask=src_mask,
                                                                   cond_type=cond_type,
                                                                   motion_length=motion_length,
                                                                   num_intervals=num_intervals)
        
            
            for index in range(self.copy_blocks_num + 1, self.total_blocks_num):
                
                h = self.base_model.temporal_decoder_blocks[index](x=h,
                                                                   xf=xf_out,
                                                                   emb=emb,
                                                                   src_mask=src_mask,
                                                                   cond_type=cond_type,
                                                                   motion_length=motion_length,
                                                                   num_intervals=num_intervals)
        else:
            for index in range(1, self.total_blocks_num):
                h = self.base_model.temporal_decoder_blocks[index](x=h,
                                                                   xf=xf_out,
                                                                   emb=emb,
                                                                   src_mask=src_mask,
                                                                   cond_type=cond_type,
                                                                   motion_length=motion_length,
                                                                   num_intervals=num_intervals)

        output = self.base_model.out(h).view(B, T, -1).contiguous()
        return output

    def forward_test(self,
                     h=None,
                     src_mask=None,
                     emb=None,
                     xf_out=None,
                     timesteps=None,
                     motion_length=None,
                     num_intervals=1,
                     c=None,
                     **kwargs):

        B, T = h.shape[0], h.shape[1]
        T = T * self.patch_size
        text_cond_type = torch.zeros(B, 1, 1).to(h.device) + 1
        none_cond_type = torch.zeros(B, 1, 1).to(h.device)

        all_cond_type = torch.cat((text_cond_type, none_cond_type), dim=0)
        h = h.repeat(2, 1, 1)
        xf_out = xf_out.repeat(2, 1, 1)
        emb = emb.repeat(2, 1)
        src_mask = src_mask.repeat(2, 1, 1)
        motion_length = motion_length.repeat(2, 1)
        
        input_h = h
        h = self.base_model.temporal_decoder_blocks[0](x=h,
                                                       xf=xf_out,
                                                       emb=emb,
                                                       src_mask=src_mask,
                                                       cond_type=all_cond_type,
                                                       motion_length=motion_length,
                                                       num_intervals=num_intervals)

        if c is not None:
            
            c = c.repeat(2, 1, 1)

            
            if self.cfg.condition_encode_cfg.condition_cfg:
                
                c = c * all_cond_type

            for index in range(1, self.copy_blocks_num + 1):
                c, c_skip = self.controlnet[index - 1](x=h,
                                                       xf=xf_out,
                                                       emb=emb,
                                                       src_mask=src_mask,
                                                       cond_type=all_cond_type,
                                                       motion_length=motion_length,
                                                       num_intervals=num_intervals,
                                                       c=c)
                h = self.base_model.temporal_decoder_blocks[index](x=h + c_skip,
                                                                   xf=xf_out,
                                                                   emb=emb,
                                                                   src_mask=src_mask,
                                                                   cond_type=all_cond_type,
                                                                   motion_length=motion_length,
                                                                   num_intervals=num_intervals)
        
            
            for index in range(self.copy_blocks_num + 1, self.total_blocks_num):
                h = self.base_model.temporal_decoder_blocks[index](x=h,
                                                                   xf=xf_out,
                                                                   emb=emb,
                                                                   src_mask=src_mask,
                                                                   cond_type=all_cond_type,
                                                                   motion_length=motion_length,
                                                                   num_intervals=num_intervals)
        else:
            for index in range(1, self.total_blocks_num):
                h = self.base_model.temporal_decoder_blocks[index](x=h,
                                                                   xf=xf_out,
                                                                   emb=emb,
                                                                   src_mask=src_mask,
                                                                   cond_type=all_cond_type,
                                                                   motion_length=motion_length,
                                                                   num_intervals=num_intervals)
        out = self.base_model.out(h).view(2 * B, T, -1).contiguous()
        out_text = out[:B].contiguous()
        out_none = out[B:].contiguous()

        coef_cfg = self.base_model.scale_func(int(timesteps[0]))
        text_coef = coef_cfg['text_coef']
        none_coef = coef_cfg['none_coef']
        output = out_text * text_coef + out_none * none_coef
        return output


    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        if all((k.startswith('base_model') or k.startswith('controlnet')) for k in state_dict.keys()):
            return super().load_state_dict(state_dict, strict)
        else:
            new_key = {}
            for k in state_dict.keys():
                new_key[k] = re.sub(r"(temporal_decoder_blocks\.\d+)(.*)", r"\1.base_block\2", k)
            for k, v in new_key.items():
                if k != v:
                    print(f"replace {k} to {v}")
                    state_dict[v] = state_dict.pop(k)

            return self.base_model.load_state_dict(state_dict, strict)

    def aux_loss(self):
        aux_loss = 0
        kl_loss = 0
        for module in self.controlnet:
            if hasattr(module.copied_block.ca_block, 'aux_loss'):
                aux_loss = aux_loss + module.copied_block.ca_block.aux_loss
            if hasattr(module.copied_block.ca_block, 'kl_loss'):
                kl_loss = kl_loss + module.copied_block.ca_block.kl_loss
        losses = {}
        if aux_loss > 0:
            losses['moe_route_loss'] = aux_loss * self.base_model.moe_route_loss_weight
        if kl_loss > 0:
            losses['template_kl_loss'] = kl_loss * self.base_model.template_kl_loss_weight
        return losses
    
    def get_precompute_condition(self,
                                 **kwargs):
        return self.base_model.get_precompute_condition(**kwargs)
    def post_process(self,
                     output):
        return self.base_model.post_process(output)
    
    @property
    def dtype(self):
        
        return next(self.parameters()).dtype