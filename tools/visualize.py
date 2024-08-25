
import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ["DISPLAY"] = ':0.0'
import warnings
warnings.filterwarnings("ignore")
import argparse
import mmcv
import numpy as np
import pandas as pd
import time
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from scipy.ndimage import gaussian_filter

from mogen.models import build_architecture
from mogen.utils.plot_utils import (plot_3d_motion, plot_siamese_3d_motion,
                                    recover_from_ric, t2m_kinematic_chain)
from mogen.datasets.EMAGE_2024.utils import other_tools

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

def plot_t2m(data, motion_length, result_path, npy_path, caption):
    joints = recover_from_ric(torch.from_numpy(data).float(), 22).numpy()
    joints = motion_temporal_filter(joints, sigma=2.5)
    plot_3d_motion(save_path=result_path,
                   motion_length=motion_length,
                   kinematic_tree=t2m_kinematic_chain,
                   joints=joints,
                   title=caption,
                   fps=20)
    if npy_path is not None:
        np.save(npy_path, joints)


def plot_interhuman(data, result_path, npy_path, caption):
    data = data.reshape(data.shape[0], 2, -1)
    joints1 = data[:, 0, :22 * 3].reshape(-1, 22, 3)
    joints2 = data[:, 1, :22 * 3].reshape(-1, 22, 3)
    joints1 = motion_temporal_filter(joints1, sigma=4.5)
    joints2 = motion_temporal_filter(joints2, sigma=4.5)
    plot_siamese_3d_motion(save_path=result_path,
                           kinematic_tree=t2m_kinematic_chain,
                           mp_joints=[joints1, joints2],
                           title=caption,
                           fps=30)

def plot_t2m_smplx(
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
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    # build the model and load checkpoint
    cfg.model['opt'] = args
    model = build_architecture(cfg.model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')

    if args.device == 'cpu':
        model = model.cpu()
    else:
        model = MMDataParallel(model, device_ids=[0])
    model.eval()

    dataset_name = cfg.data.test.dataset_name
    assert dataset_name in ["human_ml3d", "inter_human", "motionx"]
    assert len(args.motion_length) == len(args.text)
    max_length = max(args.motion_length)
    if dataset_name == "human_ml3d":
        input_dim = 263
        assert max_length >= 16 and max_length <= 196
    elif dataset_name == "inter_human":
        input_dim = 524
        assert max_length >= 16 and max_length <= 300
    elif dataset_name == "motionx":
        input_dim = 322
        assert max_length >= 64 and max_length <= 196
    try:
        if dataset_name == "motionx":
            mean_path = os.path.join("./data", "datasets", dataset_name, "humanml3d_align_mean.npy")
            std_path = os.path.join("./data", "datasets", dataset_name, "humanml3d_align_std.npy")
        else:
            mean_path = os.path.join("./data", "datasets", dataset_name, "mean.npy")
            std_path = os.path.join("./data", "datasets", dataset_name, "std.npy")
        mean = np.load(mean_path)
        std = np.load(std_path)
    except Exception as e:
        print(f"{mean_path} or {std_path} not exists! Employ the default values: 0 and 1")
        mean = np.zeros((input_dim))
        std = np.ones((input_dim))

    device = args.device
    num_intervals = len(args.text)
    motion = torch.zeros(num_intervals, max_length, input_dim).to(device)
    motion_mask = torch.zeros(num_intervals, max_length).to(device)
    for i in range(num_intervals):
        motion_mask[i, :args.motion_length[i]] = 1
    motion_length = torch.Tensor(args.motion_length).long().to(device)
    model = model.to(device)
    metas = []
    
    for t in args.text:
        metas.append({'text': t})
        print(f"Text: {t}")
        
    input = {
        'motion': motion,
        'motion_mask': motion_mask,
        'motion_length': motion_length,
        'num_intervals': num_intervals,
        'motion_metas': metas,
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


    print(f'pred_motion: {pred_motion.shape}') # (T, D)

    if dataset_name == "human_ml3d":
        plot_t2m(data=pred_motion,
                 motion_length=args.motion_length,
                 result_path=args.out,
                 npy_path=args.pose_npy,
                 caption=args.text)
    elif dataset_name == "inter_human":
        plot_interhuman(data=pred_motion,
                        result_path=args.out,
                        npy_path=args.pose_npy,
                        caption=args.text)
    elif dataset_name == "motionx":
        save_path = args.out 

        T= pred_motion.shape[0]
        betas_np = np.zeros((300))

        rec_pose_np = np.zeros((T, 165))
        rec_pose_np[:, :3+63] = pred_motion[:, :3+63]
        rec_pose_np[:, 66:66+3] = pred_motion[:, 66+90:66+93]
        rec_pose_np[:, 66+9:66+90+9] = pred_motion[:, 66:66+90]

        rec_trans_np = pred_motion[:, 309:309+3]
        rec_exp_np = pred_motion[:, 209:209+100]
        rec_pose_np = motion_temporal_filter_wo_reshape(rec_pose_np, sigma=3.5)
        rec_trans_np = motion_temporal_filter_wo_reshape(rec_trans_np, sigma=3.0)
        rec_exp_np = motion_temporal_filter_wo_reshape(rec_exp_np, sigma=2.0)
        np.savez(os.path.join(save_path, "res_" + args.text[0].replace('/', '_').replace(' ', '_').replace('.', '')) + f"_{int(motion_length[0])}" + '.npz',
            betas=betas_np,
            poses=rec_pose_np,
            expressions=rec_exp_np,
            trans=rec_trans_np,
            # trans=gt_trans_np,
            model='smplx2020',
            gender='neutral',
            mocap_frame_rate = 30,
        )
        plot_t2m_smplx(
            res_name = "res_" + args.text[0].replace('/', '_').replace(' ', '_').replace('.', '') + f"_{int(motion_length[0])}",
            save_path = save_path,
            smplx_models_path = './data/datasets/beats2/PantoMatrix/EMAGE/'
        )
        print("Generation finishes.")


if __name__ == '__main__':
    main()