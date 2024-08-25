import argparse
import os

os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ["DISPLAY"] = ':0.0'
import warnings
warnings.filterwarnings("ignore")
import mmcv
import numpy as np
import pandas as pd
import time
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from scipy.ndimage import gaussian_filter

from mogen.models import build_architecture
from mogen.utils.plot_utils import (plot_3d_motion, t2m_kinematic_chain)
from mogen.datasets.EMAGE_2024.utils import other_tools

from mogen.models.transformers.controlnet import ControlT2MHalf
from mogen.models.transformers.controlnet_mcm import ControlT2MHalf_MCM

from mogen.models.utils.quaternion import ax_from_6v, ax_to_6v


import json
from tqdm import tqdm

class Beats2Args:
    def __init__(self) -> None:
        self.debug = False
        self.render_video_fps = 30
        self.render_video_width = 1920
        self.render_video_height = 720
        self.render_concurrent_num = 8
        self.render_tmp_img_filetype = "bmp"

def motion_temporal_filter(motion, sigma=1):
    motion = motion.reshape(motion.shape[0], -1)
    for i in range(motion.shape[1]):
        motion[:, i] = gaussian_filter(motion[:, i],
                                       sigma=sigma,
                                       mode="nearest")
    return motion.reshape(motion.shape[0], -1, 3)

def motion_temporal_filter_wo_reshape(motion, sigma=1):
    for i in range(motion.shape[1]):
        motion[:, i] = gaussian_filter(motion[:, i],
                                       sigma=sigma,
                                       mode="nearest")
    return motion

def plot_skeleton(data, motion_length, result_path, npy_path, caption):
    
    joints = motion_temporal_filter(data, sigma=2.5)
    plot_3d_motion(save_path=result_path,
                   motion_length=motion_length,
                   kinematic_tree=t2m_kinematic_chain,
                   joints=joints,
                   title=caption,
                   fps=30)
    if npy_path is not None:
        np.save(npy_path, joints)


def plot_smplx(
        res_name,
        save_path='./samples/motionx',
        smplx_models_path='./EMAGE/',
        ):
    args = Beats2Args()
    start_time = time.time()
    other_tools.render_one_sequence_wo_gt(
        os.path.join(save_path, res_name)+'.npz',  
        save_path,
        smplx_models_path+"smplx_models/", 
        use_matplotlib = False,
        args = args,
        )
    end_time = time.time() - start_time
    print(f"total inference time: {int(end_time)} s")

def parse_args():
    parser = argparse.ArgumentParser(description='mogen evaluation')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--text', help='motion description', nargs='+')
    parser.add_argument('--slice_idx', type=int, default=0, help='The generated slice idx')
    parser.add_argument('--motion_length',
                        type=int,
                        help='expected motion length',
                        nargs='+')
    parser.add_argument('--out', help='output animation file')
    parser.add_argument('--repaint', action='store_true', help='whether to use repaint for a long sequence')
    parser.add_argument('--overlap_len', type=int, default=0, help='Fix the initial N frames for this clip')
    parser.add_argument('--fix_very_first', action='store_true', help='Fix the very first {overlap_len} frames for this video to be the same as GT')
    parser.add_argument('--same_overlap_noisy', action="store_true", help='During the outpainting process, use the same overlapping noisyGT')
    parser.add_argument('--no_resample', action="store_true", help='Do not use resample during inpainting based sampling')
    parser.add_argument("--timestep_respacing", type=str, default='ddim1000', help="Set ddim steps 'ddim{STEP}'")
    parser.add_argument('--jump_n_sample', type=int, default=5, help='hyperparameter for resampling')
    parser.add_argument('--jump_length', type=int, default=3, help='hyperparameter for resampling')
    parser.add_argument('--addBlend', type=bool, default=True, help='Blend in the overlapping region at the last two denoise steps')
    parser.add_argument('--no_repaint', action="store_true", help='Do not perform repaint during long-form generation')
    parser.add_argument('--pose_npy',
                        help='output pose sequence file',
                        default=None)
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--device',
                        choices=['cpu', 'cuda'],
                        default='cuda',
                        help='device used for testing')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def get_windows(x, size, step):
    if isinstance(x, dict):
        out = {}
        for key in x.keys():
            out[key] = get_windows(x[key], size, step)
        out_dict_list = []
        for i in range(len(out[list(out.keys())[0]])):
            out_dict_list.append({key: out[key][i] for key in out.keys()})
        return out_dict_list
    else:
        seq_len = x.shape[1]
        if seq_len <= size:
            return [x]
        else:
            win_num = (seq_len - (size-step)) / float(step)
            out = [x[:, mm*step : mm*step + size, ...] for mm in range(int(win_num))]
            if win_num - int(win_num) != 0:
                out.append(x[:, int(win_num)*step:, ...])  
            return out

def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    
    cfg.model['opt'] = args
    model = build_architecture(cfg.model)

    print(f'Adding control branch for {cfg.model.model.type}')
    if cfg.model.model.type == 'MCMTransformer':
        control_net = ControlT2MHalf_MCM(model.model, copy_blocks_num=cfg.copy_blocks_num, control_cond_feats=cfg.control_cond_feats, cfg=cfg).train()
    elif cfg.model.model.type == 'STMoGenTransformer':
        control_net = ControlT2MHalf(model.model, copy_blocks_num=cfg.copy_blocks_num, control_cond_feats=cfg.control_cond_feats, cfg=cfg).train()
    else:
        raise NotImplementedError   

    model.model = control_net
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.device == 'cpu':
        model = model.cpu()
    else:
        model = MMDataParallel(model, device_ids=[0])
    model.eval()

    dataset_name = cfg.data.test.dataset_name
    assert dataset_name in ["finedance"]

    max_length = max(args.motion_length)
    if dataset_name == "finedance":
        input_dim = 322
        assert max_length >= 120 and max_length <= 196
        mean_path = os.path.join("./data", "datasets", dataset_name, "mean.npy")
        std_path = os.path.join("./data", "datasets", dataset_name, "std.npy")
        mean = np.load(mean_path)
        std = np.load(std_path)
    else:
        raise NotImplementedError

    device = args.device
    num_intervals = len(args.text)
    motion = torch.zeros(num_intervals, max_length, input_dim).to(device)
    motion_mask = torch.zeros(num_intervals, max_length).to(device)
    for i in range(num_intervals):
        motion_mask[i, :args.motion_length[i]] = 1
    motion_length = torch.Tensor(args.motion_length).long().to(device)
    model = model.to(device)
    metas = []
    if dataset_name == 'finedance':
        text_path = f'./data/datasets/finedance/label_json/{args.text[0]}.json'

        with open(text_path, 'r') as file:
            json_text = json.load(file)
            text_data = f"A dancer is performing a {json_text['style1']} dance in the {json_text['style2']} style to the rhythm of the {json_text['name']} song."
        
        print(f'Text Guidance: {text_data}')
        metas.append({'text': text_data})

        music_path = f'./data/datasets/finedance/music_npy/{args.text[0]}.npy'
        music_data = torch.Tensor(np.load(music_path)).to(device)
        
        before_offset = 360
        music_data = music_data[before_offset:]
        music_data = music_data[args.slice_idx*motion_length[0] : (args.slice_idx+1) * motion_length[0]]
    
    if args.repaint == False:
        input = {
            'motion': motion,
            'motion_mask': motion_mask,
            'motion_length': motion_length,
            'num_intervals': num_intervals,
            'motion_metas': metas,
            'c': music_data.unsqueeze(0),
        }
        all_pred_motion = []
        with torch.no_grad():
            input['inference_kwargs'] = {}
            output = model(**input)
            for i in range(num_intervals):
                pred_motion = output[i]['pred_motion'][:int(motion_length[i])]
                pred_motion = pred_motion.cpu().detach().numpy()
                pred_motion = pred_motion * std + mean
                all_pred_motion.append(pred_motion)
            pred_motion = np.concatenate(all_pred_motion, axis=0)
    
    print(f'pred_motion: {pred_motion.shape}') 

    if dataset_name == "finedance":

        save_path = args.out 
        T= pred_motion.shape[0]
        betas_np = np.zeros((300))

        rec_pose_np = np.zeros((T, 165))
        rec_pose_np[:, :3+63] = pred_motion[:, :3+63]
        rec_pose_np[:, 66+9:66+90+9] = pred_motion[:, 66:66+90]
        rec_trans_np = pred_motion[:, 309:309+3]
        rec_trans_np[:, 1] = rec_trans_np[:, 1]
        rec_exp_np = np.zeros((T, 100))

        
        rec_trans_np = motion_temporal_filter_wo_reshape(rec_trans_np, sigma=3.0)
        rec_pose_np[:, :3+63] = ax_from_6v(
                    torch.Tensor(
                        motion_temporal_filter_wo_reshape(
                            ax_to_6v(
                                torch.Tensor(rec_pose_np[:, :3+63]).reshape(T, 22, 3)
                                ).numpy().reshape(T, -1)
                                , sigma=3.0)
                        ).reshape(T, 22, 6)
                    ).reshape(T, 66)
        rec_pose_np[:, 66+9:66+90+9] = ax_from_6v(
            torch.Tensor(
                motion_temporal_filter_wo_reshape(
                    ax_to_6v(
                        torch.Tensor(rec_pose_np[:, 66+9:66+90+9]).reshape(T, 30, 3)
                        ).numpy().reshape(T, -1)
                        , sigma=3.0)
                ).reshape(T, 30, 6)
            ).reshape(T, 90)

        np.savez(os.path.join(save_path, "res_" + args.text[0]) + f"_{int(args.motion_length[0])}_{args.slice_idx}_{args.checkpoint.split('/')[-1].split('.')[0]}" + '.npz',
            betas=betas_np,
            poses=rec_pose_np,
            expressions=rec_exp_np,
            trans=rec_trans_np,
            model='smplx2020',
            gender='neutral',
            mocap_frame_rate = 30,
        )
        plot_smplx(
            res_name = "res_" + args.text[0] + f"_{int(args.motion_length[0])}_{args.slice_idx}_{args.checkpoint.split('/')[-1].split('.')[0]}",
            save_path = save_path,
            smplx_models_path = './data/datasets/beats2/PantoMatrix/EMAGE/'
        )


if __name__ == '__main__':
    main()
