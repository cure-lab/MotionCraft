import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import ATTENTIONS
from ..utils.stylization_block import StylizationBlock

from .efficient_attention import EfficientSelfAttention

try:
    from tutel import moe as tutel_moe
    from tutel import net
except ImportError:
    pass


class MOE(nn.Module):

    def __init__(self, num_experts, topk, input_dim, ffn_dim, output_dim,
                 num_heads, max_seq_len, gate_type, gate_noise):
        super().__init__()
        self.proj = nn.Linear(input_dim, output_dim)
        self.activation = nn.GELU()
        try:
            data_group = net.create_groups_from_world(group_count=1).data_group
        except:
            data_group = None
        self.model = tutel_moe.moe_layer(gate_type={
            'type': gate_type,
            'k': topk,
            'fp32_gate': True,
            'gate_noise': gate_noise,
            'capacity_factor': 1.5
        },
                                         experts={
                                             'type': 'ffn',
                                             'count_per_node': num_experts,
                                             'hidden_size_per_expert': ffn_dim,
                                             'activation_fn':
                                             lambda x: F.gelu(x)
                                         },
                                         model_dim=input_dim,
                                         batch_prioritized_routing=True,
                                         is_gshard_loss=False,
                                         group=data_group)
        self.embedding = nn.Parameter(
            torch.randn(1, max_seq_len, num_heads, input_dim))

    def forward(self, x):
        B, T, H, D = x.shape
        x = x + self.embedding[:, :T, :, :]
        x = x.reshape(-1, D)
        y = self.proj(self.activation(self.model(x)))
        self.aux_loss = self.model.l_aux
        y = y.reshape(B, T, H, -1)
        return y


def get_ffn(latent_dim, ffn_dim):
    return nn.Sequential(nn.Linear(latent_dim, ffn_dim), nn.GELU(),
                         nn.Linear(ffn_dim, latent_dim))


@ATTENTIONS.register_module()
class STMA(nn.Module):

    def __init__(self, latent_dim, text_latent_dim, num_heads, num_text_heads,
                 num_experts, topk, gate_type, gate_noise, ffn_dim,
                 time_embed_dim, max_seq_len, max_text_seq_len, temporal_comb,
                 dropout, static_body=True, dynamic_body=False, patch_size=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.num_text_heads = num_text_heads
        self.max_seq_len = max_seq_len
        self.patch_size = patch_size

        self.norm = nn.LayerNorm(latent_dim)
        self.text_norm = nn.LayerNorm(text_latent_dim)

        self.text_moe = MOE(num_experts, topk, text_latent_dim,
                            text_latent_dim * 4, 2 * latent_dim,
                            num_text_heads, max_text_seq_len, gate_type,
                            gate_noise)
        self.motion_moe = MOE(num_experts, topk, latent_dim, latent_dim * 4,
                              4 * latent_dim, num_heads, max_seq_len,
                              gate_type, gate_noise)
        self.body_weight = nn.Parameter(torch.randn(num_heads, num_heads))
        self.static_body = static_body
        self.dynamic_body = dynamic_body
        if self.dynamic_body:
            self.body_d_attn = EfficientSelfAttention(
                latent_dim=latent_dim,
                
                num_heads=8,  
                dropout=dropout,
                time_embed_dim=None
            )

        self.proj_out = StylizationBlock(latent_dim * num_heads,
                                         time_embed_dim, dropout)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, xf, emb, src_mask, cond_type, motion_length,
                num_intervals, **kwargs):
        """
        x: B, T, D
        xf: B, N, P
        """
        B, T, D = x.shape
        N = xf.shape[1] + x.shape[1]
        H = self.num_heads
        L = self.latent_dim

        x = x.reshape(B, T, H, -1)
        
        text_feat = xf.reshape(B, xf.shape[1], self.num_text_heads, -1)
        text_feat = self.text_moe(self.text_norm(text_feat))
        motion_feat = self.motion_moe(self.norm(x))

        
        body_weight = F.softmax(self.body_weight, dim=1)
        body_value = motion_feat[:, :, :, :L]
        body_feat = body_value
        if self.static_body:
            body_feat = torch.einsum('hl,bnld->bnhd', body_weight, body_value)
        body_feat = body_feat.reshape(B, T, D)
        if self.dynamic_body:
            
            d_body_feat = self.body_d_attn(body_value.reshape(B*T, H, -1), torch.ones((B*T, H, 1)).to(body_value.device)).reshape(B, T, D)
            
            
            body_feat = body_feat + d_body_feat

        
        text_cond_type = (cond_type % 10 > 0).float().unsqueeze(-1)
        if self.patch_size > 1:
            
            src_mask = src_mask.squeeze(-1)
            patches = src_mask.unfold(1, self.patch_size, self.patch_size)
            has_zero = (patches == 0).any(dim=2)
            src_mask = ~has_zero
            src_mask = src_mask.float()
            
        src_mask = src_mask.view(B, T, 1, 1)

        key_text = text_feat[:, :, :, :L].contiguous()
        key_text = key_text + (1 - text_cond_type) * -1000000
        if self.num_text_heads == 1:
            key_text = key_text.repeat(1, 1, H, 1)
        key_motion = motion_feat[:, :, :, L:2 * L].contiguous()
        key_motion = key_motion + (1 - src_mask) * -1000000
        key = torch.cat((key_text, key_motion), dim=1)
        key = F.softmax(key.view(B, N, H, -1), dim=1)
        

        value_text = text_feat[:, :, :, L:].contiguous() * text_cond_type
        if self.num_text_heads == 1:
            value_text = value_text.repeat(1, 1, H, 1)
        value_motion = motion_feat[:, :, :, 2 * L: 3 * L].contiguous() * src_mask
        value = torch.cat((value_text, value_motion), dim=1).view(B, N, H, -1)

        query = motion_feat[:, :, :, 3 * L:].contiguous()
        query = F.softmax(query.view(B, T, H, -1), dim=-1)

        
        attention = torch.einsum('bnhd,bnhl->bhdl', key, value)

        y_t = torch.einsum('bnhd,bhdl->bnhl', query, attention).reshape(B, T, D)
        y_s = body_feat
        y = x.reshape(B, T, D) + self.proj_out(y_s + y_t, emb)
        if self.training:
            self.aux_loss = self.text_moe.aux_loss + self.motion_moe.aux_loss
            
            
            
            
        return y

