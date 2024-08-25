import numpy as np
import torch
from torch import nn
from einops import rearrange
from mogen.models.utils.misc import zero_module

from ..builder import SUBMODULES, build_attention
from ..utils.stylization_block import StylizationBlock
from .diffusion_transformer import DiffusionTransformer
from ..gnns.stgcn import STGCN
from ..utils.blocks import PatchEmbed1D

def get_kit_slice(idx):
    if idx == 0:
        result = [0, 1, 2, 3, 184, 185, 186, 247, 248, 249, 250]
    else:
        result = [
            4 + (idx - 1) * 3,
            4 + (idx - 1) * 3 + 1,
            4 + (idx - 1) * 3 + 2,
            64 + (idx - 1) * 6,
            64 + (idx - 1) * 6 + 1,
            64 + (idx - 1) * 6 + 2,
            64 + (idx - 1) * 6 + 3,
            64 + (idx - 1) * 6 + 4,
            64 + (idx - 1) * 6 + 5,
            184 + idx * 3,
            184 + idx * 3 + 1,
            184 + idx * 3 + 2,
        ]
    return result

def get_t2m_slice(idx):
    if idx == 0:
        result = [0, 1, 2, 3, 193, 194, 195, 259, 260, 261, 262]
    else:
        result = [
            4 + (idx - 1) * 3,
            4 + (idx - 1) * 3 + 1,
            4 + (idx - 1) * 3 + 2,
            67 + (idx - 1) * 6,
            67 + (idx - 1) * 6 + 1,
            67 + (idx - 1) * 6 + 2,
            67 + (idx - 1) * 6 + 3,
            67 + (idx - 1) * 6 + 4,
            67 + (idx - 1) * 6 + 5,
            193 + idx * 3,
            193 + idx * 3 + 1,
            193 + idx * 3 + 2,
        ]
    return result

def get_smplx_slice(idx):
    slice_dict = {
        "root": [0, 1, 2]  + [i for i in range(312, 322)],
        "trans": [309, 310, 311],
        "head": [12*3, 12*3+1, 12*3+2] + [15*3, 15*3+1, 15*3+2] + [66+90, 66+90+1, 66+90+2],
        "stem": [3*3, 3*3+1, 3*3+2] + [6*3, 6*3+1, 6*3+2] + [9*3, 9*3+1, 9*3+2],
        "larm": [14*3, 14*3+1, 14*3+2] + [17*3, 17*3+1, 17*3+2] + [19*3, 19*3+1, 19*3+2] + [21*3, 21*3+1, 21*3+2],
        "rarm": [13*3, 13*3+1, 13*3+2] + [16*3, 16*3+1, 16*3+2] + [18*3, 18*3+1, 18*3+2] + [20*3, 20*3+1, 20*3+2],
        "lleg": [2*3, 2*3+1, 2*3+2] + [5*3, 5*3+1, 5*3+2] + [8*3, 8*3+1, 8*3+2] + [11*3, 11*3+1, 11*3+2],
        "rleg": [1*3, 1*3+1, 1*3+2] + [4*3, 4*3+1, 4*3+2] + [7*3, 7*3+1, 7*3+2] + [10*3, 10*3+1, 10*3+2],
        "face": [i for i in range(159, 159+100+50)],
        "lhand": [i for i in range(66, 66+45)],
        "rhand": [i for i in range(66+45, 66+90)],
    }
    result = slice_dict[str(idx)]
    return result

def get_openpose17_slice(idx):
    slice_dict = {
        "head": [i*2 for i in [0, 1, 2, 3, 4]] + [i*2+1 for i in [0, 1, 2, 3, 4]],
        "rarm": [i*2 for i in [6, 8, 10]] + [i*2+1 for i in [6, 8, 10]],
        "larm": [i*2 for i in [5, 7, 9]] + [i*2+1 for i in [5, 7, 9]],
        "rleg": [i*2 for i in [12, 14, 16]] + [i*2+1 for i in [12, 14, 16]],
        "lleg": [i*2 for i in [11, 13, 15]] + [i*2+1 for i in [11, 13, 15]],
    }
    result = slice_dict[str(idx)]
    return result

def get_rot6d_slice(idx):
    slice_dict = {
        "root": [7+0*6+0, 7+0*6+1, 7+0*6+2, 7+0*6+3, 7+0*6+4, 7+0*6+5],
        "trans": [0, 1, 2, 3] + [4, 5, 6],
        "head": [7+12*6, 7+12*6+1, 7+12*6+2, 7+12*6+3, 7+12*6+4, 7+12*6+5] + \
                [7+15*6, 7+15*6+1, 7+15*6+2, 7+15*6+3, 7+15*6+4, 7+15*6+5] + \
                [319+0, 319+1, 319+2, 319+3, 319+4, 319+5],
        "stem": [7+3*6, 7+3*6+1, 7+3*6+2, 7+3*6+3, 7+3*6+4, 7+3*6+5] + \
                [7+6*6, 7+6*6+1, 7+6*6+2, 7+6*6+3, 7+6*6+4, 7+6*6+5] + \
                [7+9*6, 7+9*6+1, 7+9*6+2, 7+9*6+3, 7+9*6+4, 7+9*6+5],
        "larm": [7+14*6, 7+14*6+1, 7+14*6+2, 7+14*6+3, 7+14*6+4, 7+14*6+5] + \
                [7+17*6, 7+17*6+1, 7+17*6+2, 7+17*6+3, 7+17*6+4, 7+17*6+5] + \
                [7+19*6, 7+19*6+1, 7+19*6+2, 7+19*6+3, 7+19*6+4, 7+19*6+5] + \
                [7+21*6, 7+21*6+1, 7+21*6+2, 7+21*6+3, 7+21*6+4, 7+21*6+5],
        "rarm": [7+13*6, 7+13*6+1, 7+13*6+2, 7+13*6+3, 7+13*6+4, 7+13*6+5] + \
                [7+16*6, 7+16*6+1, 7+16*6+2, 7+16*6+3, 7+16*6+4, 7+16*6+5] + \
                [7+18*6, 7+18*6+1, 7+18*6+2, 7+18*6+3, 7+18*6+4, 7+18*6+5] + \
                [7+20*6, 7+20*6+1, 7+20*6+2, 7+20*6+3, 7+20*6+4, 7+20*6+5],
        "lleg": [7+2*6, 7+2*6+1, 7+2*6+2, 7+2*6+3, 7+2*6+4, 7+2*6+5] + \
                [7+5*6, 7+5*6+1, 7+5*6+2, 7+5*6+3, 7+5*6+4, 7+5*6+5] + \
                [7+8*6, 7+8*6+1, 7+8*6+2, 7+8*6+3, 7+8*6+4, 7+8*6+5] + \
                [7+11*6, 7+11*6+1, 7+11*6+2, 7+11*6+3, 7+11*6+4, 7+11*6+5],
        "rleg": [7+1*6, 7+1*6+1, 7+1*6+2, 7+1*6+3, 7+1*6+4, 7+1*6+5] + \
                [7+4*6, 7+4*6+1, 7+4*6+2, 7+4*6+3, 7+4*6+4, 7+4*6+5] + \
                [7+7*6, 7+7*6+1, 7+7*6+2, 7+7*6+3, 7+7*6+4, 7+7*6+5] + \
                [7+10*6, 7+10*6+1, 7+10*6+2, 7+10*6+3, 7+10*6+4, 7+10*6+5],
        "face": [i for i in range(325, 425)],
        "lhand": [i for i in range(7+22*6, 7+22*6+15*6)],
        "rhand": [i for i in range(7+22*6+15*6, 7+22*6+30*6)],
    }
    result = slice_dict[str(idx)]
    return result

def get_part_slice(idx_list, func):
    result = []
    for idx in idx_list:
        result.extend(func(idx))
    return result

def unpatchify(x, patch_size):
        """
        Args:
            x (torch.Tensor): of shape [B, N_t, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """

        N_t = x.shape[1]
        T_p = patch_size
        out_channels = x.shape[2] // T_p
        x = rearrange(
            x,
            "B (N_t) (T_p C_out) -> B (N_t T_p) C_out",
            N_t=N_t,
            T_p=T_p,
            C_out=out_channels,
        )
        return x

class PoseEncoder(nn.Module):

    def __init__(self,
                 dataset_name="human_ml3d",
                 latent_dim=64,
                 input_dim=263,
                 patch_size=1,
                 joints=False,
                 body_graph=False,
                 gnn_cfg=None,
                ):
        super().__init__()
        self.dataset_name = dataset_name
        self.patch_size = patch_size
        self.joints = joints
        self.latent_dim = latent_dim
        if self.joints:
            if dataset_name == "human_ml3d":
                self.joints_num = 22
                func = get_t2m_slice
                self.joints_slice = [
                    get_part_slice([idx], func) for idx in range(self.joints_num)
                ]
                self.body_slice = get_part_slice([_ for _ in range(self.joints_num)], func)
            elif dataset_name == "motionx":
                self.joints_num = 57
                self.joints_slice = [
                    [i*3, i*3+1, i*3+2] for i in range(52+1)
                ]
                self.joints_slice.append([309, 310, 311])
                self.joints_slice.append([_ for _ in range(159, 159+50)])
                self.joints_slice.append([_ for _ in range(209, 209+100)])
                self.joints_slice.append([_ for _ in range(312, 322)])
                self.body_slice = [i for i in range(66+93)] + [309, 310, 311] + [_ for _ in range(159, 159+50)] + [_ for _ in range(209, 209+100)] + [_ for _ in range(312, 322)]
            else:
                raise ValueError()
            
            self.linear_list = nn.ModuleList()
            for i in range(len(self.joints_slice)):
                self.linear_list.append(
                    nn.Linear(len(self.joints_slice[i]), latent_dim)
                )
            
            self.body_embed = nn.Linear(len(self.body_slice), latent_dim)
            assert len(set(self.body_slice)) == input_dim
            self.parts_num = self.joints_num + 1 
        else:
            if dataset_name == "human_ml3d":
                self.parts_num = 8
                func = get_t2m_slice
                self.head_slice = get_part_slice([12, 15], func)
                self.stem_slice = get_part_slice([3, 6, 9], func)
                self.larm_slice = get_part_slice([14, 17, 19, 21], func)
                self.rarm_slice = get_part_slice([13, 16, 18, 20], func)
                self.lleg_slice = get_part_slice([2, 5, 8, 11], func)
                self.rleg_slice = get_part_slice([1, 4, 7, 10], func)
                self.root_slice = get_part_slice([0], func)
                self.body_slice = get_part_slice([_ for _ in range(22)], func)
            elif dataset_name == "kit_ml":
                self.parts_num = 8
                func = get_kit_slice
                self.head_slice = get_part_slice([4], func)
                self.stem_slice = get_part_slice([1, 2, 3], func)
                self.larm_slice = get_part_slice([8, 9, 10], func)
                self.rarm_slice = get_part_slice([5, 6, 7], func)
                self.lleg_slice = get_part_slice([16, 17, 18, 19, 20], func)
                self.rleg_slice = get_part_slice([11, 12, 13, 14, 15], func)
                self.root_slice = get_part_slice([0], func)
                self.body_slice = get_part_slice([_ for _ in range(21)], func)
            elif dataset_name == "motionx" or dataset_name == "rot6d":
                self.parts_num = 12
                func = get_smplx_slice if dataset_name == "motionx" else get_rot6d_slice
                self.head_slice = get_part_slice(['head'], func)
                self.stem_slice = get_part_slice(['stem'], func)
                self.larm_slice = get_part_slice(['larm'], func)
                self.rarm_slice = get_part_slice(['rarm'], func)
                self.lleg_slice = get_part_slice(['lleg'], func)
                self.rleg_slice = get_part_slice(['rleg'], func)
                self.root_slice = get_part_slice(['root'], func)
                self.trans_slice = get_part_slice(['trans'], func)
                self.face_slice = get_part_slice(['face'], func)
                self.lhand_slice = get_part_slice(['lhand'], func)
                self.rhand_slice = get_part_slice(['rhand'], func)
                self.body_slice = get_part_slice(['head', 'stem', 'larm', 'rarm', 'lleg', 'rleg', 'root', 'trans', 'face', 'lhand', 'rhand'], func)
                if self.patch_size > 1:
                    self.trans_embed = PatchEmbed1D(
                        patch_size=(self.patch_size,),
                        in_chans=len(self.trans_slice), 
                        embed_dim=latent_dim
                        )
                    self.face_embed = PatchEmbed1D(
                        patch_size=(self.patch_size,),
                        in_chans=len(self.face_slice), 
                        embed_dim=latent_dim
                        )
                    self.lhand_embed = PatchEmbed1D(
                        patch_size=(self.patch_size,),
                        in_chans=len(self.lhand_slice), 
                        embed_dim=latent_dim
                        )
                    self.rhand_embed = PatchEmbed1D(
                        patch_size=(self.patch_size,),
                        in_chans=len(self.rhand_slice), 
                        embed_dim=latent_dim
                        )
                else:
                    self.trans_embed = nn.Linear(len(self.trans_slice), latent_dim)
                    self.face_embed = nn.Linear(len(self.face_slice), latent_dim)
                    self.lhand_embed = nn.Linear(len(self.lhand_slice), latent_dim)
                    self.rhand_embed = nn.Linear(len(self.rhand_slice), latent_dim)
            elif dataset_name == "openpose17":
                self.parts_num = 6
                func = get_openpose17_slice
                self.head_slice = get_part_slice(['head'], func)
                self.larm_slice = get_part_slice(['larm'], func)
                self.rarm_slice = get_part_slice(['rarm'], func)
                self.lleg_slice = get_part_slice(['lleg'], func)
                self.rleg_slice = get_part_slice(['rleg'], func)
                self.body_slice = get_part_slice(['head', 'larm', 'rarm', 'lleg', 'rleg'], func)
            else:
                raise NotImplementedError
            if self.patch_size > 1:
                if dataset_name != "openpose17":
                    self.head_embed = PatchEmbed1D(
                        patch_size=(self.patch_size,),
                        in_chans=len(self.head_slice), 
                        embed_dim=latent_dim
                        )
                    self.stem_embed = PatchEmbed1D(
                        patch_size=(self.patch_size,),
                        in_chans=len(self.stem_slice), 
                        embed_dim=latent_dim
                        )
                    self.larm_embed = PatchEmbed1D(
                        patch_size=(self.patch_size,),
                        in_chans=len(self.larm_slice), 
                        embed_dim=latent_dim
                        )
                    self.rarm_embed = PatchEmbed1D(
                        patch_size=(self.patch_size,),
                        in_chans=len(self.rarm_slice), 
                        embed_dim=latent_dim
                        )
                    self.lleg_embed = PatchEmbed1D(
                        patch_size=(self.patch_size,),
                        in_chans=len(self.lleg_slice), 
                        embed_dim=latent_dim
                        )
                    self.rleg_embed = PatchEmbed1D(
                        patch_size=(self.patch_size,),
                        in_chans=len(self.rleg_slice), 
                        embed_dim=latent_dim
                        )
                    self.root_embed = PatchEmbed1D(
                        patch_size=(self.patch_size,),
                        in_chans=len(self.root_slice), 
                        embed_dim=latent_dim
                        )
                    self.body_embed = PatchEmbed1D(
                        patch_size=(self.patch_size,),
                        in_chans=len(self.body_slice), 
                        embed_dim=latent_dim
                        )
            else:
                if dataset_name != "openpose17":
                    self.head_embed = nn.Linear(len(self.head_slice), latent_dim)
                    self.stem_embed = nn.Linear(len(self.stem_slice), latent_dim)
                    self.larm_embed = nn.Linear(len(self.larm_slice), latent_dim)
                    self.rarm_embed = nn.Linear(len(self.rarm_slice), latent_dim)
                    self.lleg_embed = nn.Linear(len(self.lleg_slice), latent_dim)
                    self.rleg_embed = nn.Linear(len(self.rleg_slice), latent_dim)
                    self.root_embed = nn.Linear(len(self.root_slice), latent_dim)
                    self.body_embed = nn.Linear(len(self.body_slice), latent_dim)
                else:
                    self.head_embed = nn.Linear(len(self.head_slice), latent_dim)
                    self.larm_embed = nn.Linear(len(self.larm_slice), latent_dim)
                    self.rarm_embed = nn.Linear(len(self.rarm_slice), latent_dim)
                    self.lleg_embed = nn.Linear(len(self.lleg_slice), latent_dim)
                    self.rleg_embed = nn.Linear(len(self.rleg_slice), latent_dim)
                    self.body_embed = nn.Linear(len(self.body_slice), latent_dim)

            assert len(set(self.body_slice)) == input_dim
        if body_graph:
            self.gnn = STGCN(**gnn_cfg)
        else:
            self.gnn = lambda x: x
    def forward(self, motion):
        if self.joints:
            feat_list = []
            for i in range(self.joints_num):
                feat_list.append(self.linear_list[i](motion[:, :, self.joints_slice[i]].contiguous()))
            body_feat = self.body_embed(motion[:, :, self.body_slice].contiguous())
            feat_list.append(body_feat)
            feat = torch.cat(feat_list, dim=-1)
        else:
            if self.dataset_name == 'motionx' or self.dataset_name == "rot6d":
                head_feat = self.head_embed(motion[:, :, self.head_slice].contiguous())
                stem_feat = self.stem_embed(motion[:, :, self.stem_slice].contiguous())
                larm_feat = self.larm_embed(motion[:, :, self.larm_slice].contiguous())
                rarm_feat = self.rarm_embed(motion[:, :, self.rarm_slice].contiguous())
                lleg_feat = self.lleg_embed(motion[:, :, self.lleg_slice].contiguous())
                rleg_feat = self.rleg_embed(motion[:, :, self.rleg_slice].contiguous())
                root_feat = self.root_embed(motion[:, :, self.root_slice].contiguous())
                trans_feat = self.trans_embed(motion[:, :, self.trans_slice].contiguous())
                face_feat = self.face_embed(motion[:, :, self.face_slice].contiguous())
                lhand_feat = self.lhand_embed(motion[:, :, self.lhand_slice].contiguous())
                rhand_feat = self.rhand_embed(motion[:, :, self.rhand_slice].contiguous())
                body_feat = self.body_embed(motion[:, :, self.body_slice].contiguous())
                feat = torch.cat((head_feat, stem_feat, larm_feat, rarm_feat,
                                lleg_feat, rleg_feat, root_feat, trans_feat,
                                face_feat, lhand_feat, rhand_feat,
                                body_feat),
                                dim=-1)
            elif self.dataset_name != "openpose17":
                head_feat = self.head_embed(motion[:, :, self.head_slice].contiguous())
                stem_feat = self.stem_embed(motion[:, :, self.stem_slice].contiguous())
                larm_feat = self.larm_embed(motion[:, :, self.larm_slice].contiguous())
                rarm_feat = self.rarm_embed(motion[:, :, self.rarm_slice].contiguous())
                lleg_feat = self.lleg_embed(motion[:, :, self.lleg_slice].contiguous())
                rleg_feat = self.rleg_embed(motion[:, :, self.rleg_slice].contiguous())
                root_feat = self.root_embed(motion[:, :, self.root_slice].contiguous())
                body_feat = self.body_embed(motion[:, :, self.body_slice].contiguous())
                feat = torch.cat((head_feat, stem_feat, larm_feat, rarm_feat,
                                lleg_feat, rleg_feat, root_feat, body_feat),
                                dim=-1)
            elif self.dataset_name == "openpose17":
                head_feat = self.head_embed(motion[:, :, self.head_slice].contiguous())
                larm_feat = self.larm_embed(motion[:, :, self.larm_slice].contiguous())
                rarm_feat = self.rarm_embed(motion[:, :, self.rarm_slice].contiguous())
                lleg_feat = self.lleg_embed(motion[:, :, self.lleg_slice].contiguous())
                rleg_feat = self.rleg_embed(motion[:, :, self.rleg_slice].contiguous())
                body_feat = self.body_embed(motion[:, :, self.body_slice].contiguous())
                feat = torch.cat((head_feat, larm_feat, rarm_feat,
                                lleg_feat, rleg_feat, body_feat),
                                dim=-1)
        B, T, D = feat.size()
        feat = self.gnn(feat.reshape(B, T, self.parts_num, self.latent_dim)).reshape(B, T, D)
        return feat


class PoseDecoder(nn.Module):

    def __init__(self,
                 dataset_name="human_ml3d",
                 latent_dim=64,
                 output_dim=263,
                 patch_size=1,
                 joints=False):
        super().__init__()
        self.dataset_name = dataset_name
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.patch_size = patch_size
        self.joints = joints
        if self.joints:
            if dataset_name == "human_ml3d":
                self.joints_num = 22
                func = get_t2m_slice
                self.joints_slice = [
                    get_part_slice([idx], func) for idx in range(22)
                ]
                self.body_slice = get_part_slice([_ for _ in range(22)], func)
            elif dataset_name == "motionx":
                self.joints_num = 57
                self.joints_slice = [
                    [i*3, i*3+1, i*3+2] for i in range(52+1)
                ]
                self.joints_slice.append([309, 310, 311])
                self.joints_slice.append([_ for _ in range(159, 159+50)])
                self.joints_slice.append([_ for _ in range(209, 209+100)])
                self.joints_slice.append([_ for _ in range(312, 322)])
                self.body_slice = [i for i in range(66+93)] + [309, 310, 311] + [_ for _ in range(159, 159+50)] + [_ for _ in range(209, 209+100)] + [_ for _ in range(312, 322)]
            else:
                raise ValueError()
            
            self.linear_list = nn.ModuleList()
            for i in range(len(self.joints_slice)):
                self.linear_list.append(
                    nn.Linear(latent_dim, len(self.joints_slice[i]))
                )
            
            self.body_out = nn.Linear(latent_dim, len(self.body_slice))
            assert len(set(self.body_slice)) == output_dim
        else:
            if dataset_name == "human_ml3d":
                func = get_t2m_slice
                self.head_slice = get_part_slice([12, 15], func)
                self.stem_slice = get_part_slice([3, 6, 9], func)
                self.larm_slice = get_part_slice([14, 17, 19, 21], func)
                self.rarm_slice = get_part_slice([13, 16, 18, 20], func)
                self.lleg_slice = get_part_slice([2, 5, 8, 11], func)
                self.rleg_slice = get_part_slice([1, 4, 7, 10], func)
                self.root_slice = get_part_slice([0], func)
                self.body_slice = get_part_slice([_ for _ in range(22)], func)
            elif dataset_name == "kit_ml":
                func = get_kit_slice
                self.head_slice = get_part_slice([4], func)
                self.stem_slice = get_part_slice([1, 2, 3], func)
                self.larm_slice = get_part_slice([8, 9, 10], func)
                self.rarm_slice = get_part_slice([5, 6, 7], func)
                self.lleg_slice = get_part_slice([16, 17, 18, 19, 20], func)
                self.rleg_slice = get_part_slice([11, 12, 13, 14, 15], func)
                self.root_slice = get_part_slice([0], func)
                self.body_slice = get_part_slice([_ for _ in range(21)], func)
            elif dataset_name == "motionx" or dataset_name == "rot6d":
                self.parts_num = 12
                func = get_smplx_slice if dataset_name == "motionx" else get_rot6d_slice
                self.head_slice = get_part_slice(['head'], func)
                self.stem_slice = get_part_slice(['stem'], func)
                self.larm_slice = get_part_slice(['larm'], func)
                self.rarm_slice = get_part_slice(['rarm'], func)
                self.lleg_slice = get_part_slice(['lleg'], func)
                self.rleg_slice = get_part_slice(['rleg'], func)
                self.root_slice = get_part_slice(['root'], func)
                self.trans_slice = get_part_slice(['trans'], func)
                self.face_slice = get_part_slice(['face'], func)
                self.lhand_slice = get_part_slice(['lhand'], func)
                self.rhand_slice = get_part_slice(['rhand'], func)
                self.body_slice = get_part_slice(['head', 'stem', 'larm', 'rarm', 'lleg', 'rleg', 'root', 'trans', 'face', 'lhand', 'rhand'], func)
                
                self.trans_out = nn.Linear(latent_dim, patch_size * len(self.trans_slice))
                self.face_out = nn.Linear(latent_dim, patch_size * len(self.face_slice))
                self.lhand_out = nn.Linear(latent_dim, patch_size * len(self.lhand_slice))
                self.rhand_out = nn.Linear(latent_dim, patch_size * len(self.rhand_slice))
            elif dataset_name == "openpose17":
                self.parts_num = 6
                func = get_openpose17_slice
                self.head_slice = get_part_slice(['head'], func)
                self.larm_slice = get_part_slice(['larm'], func)
                self.rarm_slice = get_part_slice(['rarm'], func)
                self.lleg_slice = get_part_slice(['lleg'], func)
                self.rleg_slice = get_part_slice(['rleg'], func)
                self.body_slice = get_part_slice(['head', 'larm', 'rarm', 'lleg', 'rleg'], func)
            else:
                raise ValueError()
            if dataset_name != "openpose17":
                self.head_out = nn.Linear(latent_dim, patch_size * len(self.head_slice))
                self.stem_out = nn.Linear(latent_dim, patch_size * len(self.stem_slice))
                self.larm_out = nn.Linear(latent_dim, patch_size * len(self.larm_slice))
                self.rarm_out = nn.Linear(latent_dim, patch_size * len(self.rarm_slice))
                self.lleg_out = nn.Linear(latent_dim, patch_size * len(self.lleg_slice))
                self.rleg_out = nn.Linear(latent_dim, patch_size * len(self.rleg_slice))
                self.root_out = nn.Linear(latent_dim, patch_size * len(self.root_slice))
                self.body_out = nn.Linear(latent_dim, patch_size * len(self.body_slice))
            else:
                self.head_out = nn.Linear(latent_dim, patch_size * len(self.head_slice))
                self.larm_out = nn.Linear(latent_dim, patch_size * len(self.larm_slice))
                self.rarm_out = nn.Linear(latent_dim, patch_size * len(self.rarm_slice))
                self.lleg_out = nn.Linear(latent_dim, patch_size * len(self.lleg_slice))
                self.rleg_out = nn.Linear(latent_dim, patch_size * len(self.rleg_slice))
                self.body_out = nn.Linear(latent_dim, patch_size * len(self.body_slice))
            assert len(set(self.body_slice)) == output_dim
    def forward(self, motion):
        B, T = motion.shape[:2]
        T = T * self.patch_size 
        D = self.latent_dim
        if self.joints:
            output = torch.zeros(B, T, self.output_dim).type_as(motion)
            
            for i in range(self.joints_num):
                output[:, :, self.joints_slice[i]] = self.linear_list[i](motion[:, :, i * D : (i+1) * D].contiguous())
            body_feat = self.body_out(motion[:, :, self.joints_num * D:].contiguous())
            output = (output + body_feat) / 2.0
        else:
            if self.dataset_name == 'motionx' or self.dataset_name == "rot6d":
                head_feat = self.head_out(motion[:, :, :D].contiguous())
                stem_feat = self.stem_out(motion[:, :, D:2 * D].contiguous())
                larm_feat = self.larm_out(motion[:, :, 2 * D:3 * D].contiguous())
                rarm_feat = self.rarm_out(motion[:, :, 3 * D:4 * D].contiguous())
                lleg_feat = self.lleg_out(motion[:, :, 4 * D:5 * D].contiguous())
                rleg_feat = self.rleg_out(motion[:, :, 5 * D:6 * D].contiguous())
                root_feat = self.root_out(motion[:, :, 6 * D:7 * D].contiguous())
                trans_feat = self.trans_out(motion[:, :, 7 * D:8 * D].contiguous())
                face_feat = self.face_out(motion[:, :, 8 * D:9 * D].contiguous())
                lhand_feat = self.lhand_out(motion[:, :, 9 * D:10 * D].contiguous())
                rhand_feat = self.rhand_out(motion[:, :, 10 * D:11 * D].contiguous())
                body_feat = self.body_out(motion[:, :, 11 * D:].contiguous())

                head_feat = unpatchify(head_feat, self.patch_size)
                stem_feat = unpatchify(stem_feat, self.patch_size)
                larm_feat = unpatchify(larm_feat, self.patch_size)
                rarm_feat = unpatchify(rarm_feat, self.patch_size)
                lleg_feat = unpatchify(lleg_feat, self.patch_size)
                rleg_feat = unpatchify(rleg_feat, self.patch_size)
                root_feat = unpatchify(root_feat, self.patch_size)
                trans_feat = unpatchify(trans_feat, self.patch_size)
                face_feat = unpatchify(face_feat, self.patch_size)
                lhand_feat = unpatchify(lhand_feat, self.patch_size)
                rhand_feat = unpatchify(rhand_feat, self.patch_size)
                body_feat = unpatchify(body_feat, self.patch_size)
                
                output = torch.zeros(B, T, self.output_dim).type_as(motion)
                output[:, :, self.head_slice] = head_feat
                output[:, :, self.stem_slice] = stem_feat
                output[:, :, self.larm_slice] = larm_feat
                output[:, :, self.rarm_slice] = rarm_feat
                output[:, :, self.lleg_slice] = lleg_feat
                output[:, :, self.rleg_slice] = rleg_feat
                output[:, :, self.root_slice] = root_feat
                output[:, :, self.trans_slice] = trans_feat
                output[:, :, self.face_slice] = face_feat
                output[:, :, self.lhand_slice] = lhand_feat
                output[:, :, self.rhand_slice] = rhand_feat
                output = (output + body_feat) / 2.0
            elif self.dataset_name != "openpose17":
                head_feat = self.head_out(motion[:, :, :D].contiguous())
                stem_feat = self.stem_out(motion[:, :, D:2 * D].contiguous())
                larm_feat = self.larm_out(motion[:, :, 2 * D:3 * D].contiguous())
                rarm_feat = self.rarm_out(motion[:, :, 3 * D:4 * D].contiguous())
                lleg_feat = self.lleg_out(motion[:, :, 4 * D:5 * D].contiguous())
                rleg_feat = self.rleg_out(motion[:, :, 5 * D:6 * D].contiguous())
                root_feat = self.root_out(motion[:, :, 6 * D:7 * D].contiguous())
                body_feat = self.body_out(motion[:, :, 7 * D:].contiguous())
                output = torch.zeros(B, T, self.output_dim).type_as(motion)
                output[:, :, self.head_slice] = head_feat
                output[:, :, self.stem_slice] = stem_feat
                output[:, :, self.larm_slice] = larm_feat
                output[:, :, self.rarm_slice] = rarm_feat
                output[:, :, self.lleg_slice] = lleg_feat
                output[:, :, self.rleg_slice] = rleg_feat
                output[:, :, self.root_slice] = root_feat
                output = (output + body_feat) / 2.0
            elif self.dataset_name == "openpose17":
                head_feat = self.head_out(motion[:, :, :D].contiguous())
                larm_feat = self.larm_out(motion[:, :, D:2 * D].contiguous())
                rarm_feat = self.rarm_out(motion[:, :, 2 * D:3 * D].contiguous())
                lleg_feat = self.lleg_out(motion[:, :, 3 * D:4 * D].contiguous())
                rleg_feat = self.rleg_out(motion[:, :, 4 * D:5 * D].contiguous())
                body_feat = self.body_out(motion[:, :, 5 * D:6 * D].contiguous())
                output = torch.zeros(B, T, self.output_dim).type_as(motion)
                output[:, :, self.head_slice] = head_feat
                output[:, :, self.larm_slice] = larm_feat
                output[:, :, self.rarm_slice] = rarm_feat
                output[:, :, self.lleg_slice] = lleg_feat
                output[:, :, self.rleg_slice] = rleg_feat
                output = (output + body_feat) / 2.0
        
        return output


class SFFN(nn.Module):

    def __init__(self, latent_dim, ffn_dim, dropout, time_embed_dim, **kwargs):
        super().__init__()
        self.num_heads = kwargs['num_heads']
        self.linear1_list = nn.ModuleList()
        self.linear2_list = nn.ModuleList()
        for i in range(self.num_heads):
            self.linear1_list.append(nn.Linear(latent_dim, ffn_dim))
            self.linear2_list.append(nn.Linear(ffn_dim, latent_dim))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.proj_out = StylizationBlock(latent_dim * self.num_heads, time_embed_dim,
                                         dropout)

    def forward(self, x, emb, **kwargs):
        B, T, D = x.shape
        x = x.reshape(B, T, self.num_heads, -1)
        output = []
        for i in range(self.num_heads):
            feat = x[:, :, i].contiguous()
            feat = self.dropout(self.activation(self.linear1_list[i](feat)))
            feat = self.linear2_list[i](feat)
            output.append(feat)
        y = torch.cat(output, dim=-1)
        y = x.reshape(B, T, D) + self.proj_out(y, emb)
        return y


class DecoderLayer(nn.Module):

    def __init__(self, ca_block_cfg=None, ffn_cfg=None):
        super().__init__()
        self.ca_block = build_attention(ca_block_cfg)
        self.ffn = SFFN(**ffn_cfg)

    def forward(self, **kwargs):
        if self.ca_block is not None:
            x = self.ca_block(**kwargs)
            kwargs.update({'x': x})
        if self.ffn is not None:
            x = self.ffn(**kwargs)
        return x


@SUBMODULES.register_module()
class STMoGenTransformer(DiffusionTransformer):

    def __init__(self,
                 patch_size=1,
                 scale_func_cfg=None,
                 pose_encoder_cfg=None,
                 pose_decoder_cfg=None,
                 moe_route_loss_weight=1.0,
                 template_kl_loss_weight=0.0001,
                 **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.scale_func_cfg = scale_func_cfg
        self.joint_embed = PoseEncoder(**pose_encoder_cfg)
        self.out = zero_module(PoseDecoder(**pose_decoder_cfg))
        self.moe_route_loss_weight = moe_route_loss_weight
        self.template_kl_loss_weight = template_kl_loss_weight

    def build_temporal_blocks(self, sa_block_cfg, ca_block_cfg, ffn_cfg):
        self.temporal_decoder_blocks = nn.ModuleList()
        for i in range(self.num_layers):
            if isinstance(ffn_cfg, list):
                ffn_cfg_block = ffn_cfg[i]
            else:
                ffn_cfg_block = ffn_cfg
            self.temporal_decoder_blocks.append(
                DecoderLayer(ca_block_cfg=ca_block_cfg, ffn_cfg=ffn_cfg_block))

    def scale_func(self, timestep):
        scale = self.scale_func_cfg['scale']
        w = (1 - (1000 - timestep) / 1000) * scale + 1
        output = {'text_coef': w, 'none_coef': 1 - w}
        return output

    def aux_loss(self):
        aux_loss = 0
        kl_loss = 0
        for module in self.temporal_decoder_blocks:
            if hasattr(module.ca_block, 'aux_loss'):
                aux_loss = aux_loss + module.ca_block.aux_loss
            if hasattr(module.ca_block, 'kl_loss'):
                kl_loss = kl_loss + module.ca_block.kl_loss
        losses = {}
        if aux_loss > 0:
            losses['moe_route_loss'] = aux_loss * self.moe_route_loss_weight
        if kl_loss > 0:
            losses['template_kl_loss'] = kl_loss * self.template_kl_loss_weight
        return losses

    def get_precompute_condition(self,
                                 text=None,
                                 motion_length=None,
                                 xf_out=None,
                                 re_dict=None,
                                 device=None,
                                 sample_idx=None,
                                 clip_feat=None,
                                 **kwargs):
        if xf_out is None:
            xf_out = self.encode_text(text, clip_feat, device)
        output = {'xf_out': xf_out}
        return output

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
                      motion_length=None,
                      num_intervals=1,
                      **kwargs):
        B, T = h.shape[0], h.shape[1]
        T = T * self.patch_size
        cond_type = torch.randint(0, 100, size=(B, 1, 1)).to(h.device)
        for module in self.temporal_decoder_blocks:
            h = module(x=h,
                       xf=xf_out,
                       emb=emb,
                       src_mask=src_mask,
                       cond_type=cond_type,
                       motion_length=motion_length,
                       num_intervals=num_intervals)

        output = self.out(h).view(B, T, -1).contiguous()
        return output

    def forward_test(self,
                     h=None,
                     src_mask=None,
                     emb=None,
                     xf_out=None,
                     timesteps=None,
                     motion_length=None,
                     num_intervals=1,
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
        for module in self.temporal_decoder_blocks:
            h = module(x=h,
                       xf=xf_out,
                       emb=emb,
                       src_mask=src_mask,
                       cond_type=all_cond_type,
                       motion_length=motion_length,
                       num_intervals=num_intervals)
        out = self.out(h).view(2 * B, T, -1).contiguous()
        out_text = out[:B].contiguous()
        out_none = out[B:].contiguous()

        coef_cfg = self.scale_func(int(timesteps[0]))
        text_coef = coef_cfg['text_coef']
        none_coef = coef_cfg['none_coef']
        output = out_text * text_coef + out_none * none_coef
        return output
