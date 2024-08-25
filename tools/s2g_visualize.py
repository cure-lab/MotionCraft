import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ["DISPLAY"] = ':0.0'
import warnings
warnings.filterwarnings("ignore")
import argparse
import mmcv
import numpy as np
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from scipy.ndimage import gaussian_filter

from mogen.models import build_architecture
from mogen.utils.plot_utils import (plot_3d_motion, plot_siamese_3d_motion,
                                    recover_from_ric, t2m_kinematic_chain)

from mogen.models.transformers.controlnet import ControlT2MHalf
from mogen.models.transformers.controlnet_mcm import ControlT2MHalf_MCM

from mogen.datasets.EMAGE_2024.utils import other_tools
import smplx
import time 
import yaml
from addict import Dict

from mogen.datasets.EMAGE_2024.dataloaders.beat_testonly_stmogen import CustomDataset
from mogen.datasets.EMAGE_2024.dataloaders.build_vocab import Vocab
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

def motion_temporal_filter(motion, sigma=1):
    for i in range(motion.shape[1]):
        motion[:, i] = gaussian_filter(motion[:, i],
                                       sigma=sigma,
                                       mode="nearest")
    return motion

class BaseTrainer(object):
    def __init__(self, args, device, gen_cfg):
        self.gen_cfg = gen_cfg
        self.rank = device
        self.args = args
        self.test_data = CustomDataset(args, "test")
        self.test_loader = torch.utils.data.DataLoader(
            self.test_data, 
            batch_size=1,  
            shuffle=False,  
            num_workers=args.loader_workers,
            drop_last=False,
        )
        

        
        self.joints = self.test_data.joints
        
    
    def _load_data(self, dict_data):
        tar_pose_raw = dict_data["pose"]
        tar_pose = tar_pose_raw[:, :, :165].to(self.rank)
        tar_trans = dict_data["trans"].to(self.rank)
        tar_exps = dict_data["facial"].to(self.rank)
        in_audio = dict_data["audio"].to(self.rank) 
        in_word = dict_data["word"].to(self.rank)
        tar_beta = dict_data["beta"].to(self.rank)


        return {
            "in_audio": in_audio,
            "in_word": in_word,
            "tar_trans": tar_trans,
            "tar_exps": tar_exps,
            "tar_beta": tar_beta,
            "tar_pose": tar_pose,
        }
    
    def _g_test_stmogen(self, loaded_data, model):
        bs, n = loaded_data["tar_pose"].shape[0], loaded_data["tar_pose"].shape[1]
        tar_pose = loaded_data["tar_pose"][0]
        tar_beta = loaded_data["tar_beta"][0]
        in_word = loaded_data["in_word"][0]
        tar_exps = loaded_data["tar_exps"][0]
        in_audio = loaded_data["in_audio"][0]
        tar_trans = loaded_data["tar_trans"][0]


        remain = n%8
        if remain != 0:
            tar_pose = tar_pose[:-remain, :]
            tar_beta = tar_beta[:-remain, :]
            tar_trans = tar_trans[:-remain, :]
            in_word = in_word[:-remain]
            tar_exps = tar_exps[:-remain, :]
            n = n - remain

        num_intervals = 1
        input_dim = 322
        latent_all = torch.zeros(num_intervals, n, input_dim).to(self.rank)
        motion_length = torch.Tensor([self.args.pose_length]).long().to(self.rank)
        
        roundt = (n - self.args.pre_frames) // (self.args.pose_length - self.args.pre_frames)
        remain = (n - self.args.pre_frames) % (self.args.pose_length - self.args.pre_frames)
        round_l = self.args.pose_length - self.args.pre_frames
        net_out_val_list = []
        
        out_motions = []
        for i in tqdm(range(0, roundt)):
            in_word_tmp = in_word[i*(round_l):(i+1)*(round_l)+self.args.pre_frames]
            in_audio_tmp = in_audio[i*(16000//30*round_l):(i+1)*(16000//30*round_l)+16000//30*self.args.pre_frames, :]
            motion_mask = torch.ones(num_intervals, self.args.pose_length).to(self.rank)
            
            if i == 0:
                latent_all_tmp = latent_all[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames, :]
            else:
                latent_all_tmp = latent_all[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames, :]
                if self.gen_cfg.repaint == False:
                    latent_all_tmp[:, :self.args.pre_frames, :] = latent_last[:, -self.args.pre_frames:, :]
                else:
                    pass
            
            
            

            in_text_tmp = [self.test_data.lang_model.index2word[int(item)] for item in in_word_tmp]
            unique_list_str = []
            for item in in_text_tmp:
                if item not in unique_list_str and item != '': 
                    unique_list_str.append(item)
            in_text_tmp = 'A person is doing a speech, and the speech content is ' + \
                            ' '.join(unique_list_str)

            metas = []
            metas.append({'text': in_text_tmp})
            input = {
                'motion': latent_all_tmp,
                'motion_mask': motion_mask,
                'motion_length': motion_length,
                'num_intervals': num_intervals,
                'motion_metas': metas,
                'c': in_audio_tmp.unsqueeze(0),
            }
            if self.gen_cfg.repaint == True:
                inpaint_dict = {}
                gt_motion = torch.zeros((num_intervals, self.args.pose_length, input_dim))
                gt_motion[:, :, :3+63] = tar_pose[i*(round_l):(i+1)*(round_l)+self.args.pre_frames, :3+63]
                gt_motion[:, :, 66+90:66+93] = tar_pose[i*(round_l):(i+1)*(round_l)+self.args.pre_frames, 66:66+3]
                gt_motion[:, :, 66:66+90] = tar_pose[i*(round_l):(i+1)*(round_l)+self.args.pre_frames, 66+9:66+90+9]
                gt_motion[:, :, 309:309+3] = tar_trans[i*(round_l):(i+1)*(round_l)+self.args.pre_frames, :]
                gt_motion[:, :, 209:209+100] = tar_exps[i*(round_l):(i+1)*(round_l)+self.args.pre_frames, :]

                if self.gen_cfg.overlap_len > 0:
                    inpaint_dict['gt'] = torch.zeros_like(gt_motion,  device=self.rank)
                    inpaint_dict['outpainting_mask'] = torch.zeros_like(gt_motion, dtype=torch.bool,
                                                    device=self.rank)  
                    
                    if i == 0:
                        if self.gen_cfg.fix_very_first:
                            inpaint_dict['outpainting_mask'][..., :self.gen_cfg.overlap_len, :] = True  
                            inpaint_dict['gt'][:, :self.gen_cfg.overlap_len, ...] = torch.tensor(gt_motion)[:, :self.gen_cfg.overlap_len, ...]
                        else:
                            pass
                    elif i > 0:
                        inpaint_dict['outpainting_mask'][..., :self.gen_cfg.overlap_len, :] = True  
                        inpaint_dict['gt'][:, :self.gen_cfg.overlap_len, ...] = outputs[:, -self.gen_cfg.overlap_len:, ...]

                input['y'] = inpaint_dict
                

            all_pred_motion = []
            
            mean = np.load("./data/datasets/beats2/PantoMatrix/mean.npy")
            std = np.load("./data/datasets/beats2/PantoMatrix/std.npy")
            with torch.no_grad():
                input['inference_kwargs'] = {}
                output = model(**input)
                for num_interval_idx in range(num_intervals):
                    pred_motion = output[num_interval_idx]['pred_motion'][:int(motion_length[num_interval_idx])]
                    pred_motion = pred_motion.detach().cpu().numpy()
                    pred_motion = pred_motion * std + mean
                    all_pred_motion.append(pred_motion)
                pred_motion = np.concatenate(all_pred_motion, axis=0)
                outputs = torch.tensor(pred_motion).to(self.rank).unsqueeze(0)

            if i == roundt - 1:
                out_motions.append(pred_motion)
            else:
                out_motions.append(pred_motion[:round_l])

            if i == 0:
                net_out_val_list.append(pred_motion)
            else:
                net_out_val_list.append(pred_motion[self.args.pre_frames:, :])


            latent_last = torch.tensor([pred_motion])

        if self.gen_cfg.repaint == True:
            rec_motion = np.concatenate(out_motions, axis=0)
        else:
            rec_motion = np.concatenate(net_out_val_list, axis=0)
        tar_pose = tar_pose[:rec_motion.shape[0], :]
        tar_exps = tar_exps[:rec_motion.shape[0], :]
        tar_beta = tar_beta[:rec_motion.shape[0], :]
        tar_trans = tar_trans[:rec_motion.shape[0], :]
        return {
            'rec_motion': rec_motion,
            'tar_pose': tar_pose,
            'tar_exps': tar_exps,
            'tar_beta': tar_beta,
            'tar_trans': tar_trans,
        }

    def test_demo_stmogen(self, results_save_path, model):
        '''
        input audio and text, output motion
        do not calculate loss and metric
        save video
        '''
        start_time = time.time()
        total_length = 0
        test_seq_list = self.test_data.selected_file
        with torch.no_grad():
            for its, batch_data in enumerate(self.test_loader):
                loaded_data = self._load_data(batch_data)  
                net_out = self._g_test_stmogen(loaded_data, model)
                rec_motion = net_out['rec_motion']
                tar_pose = net_out['tar_pose']
                tar_exps = net_out['tar_exps']
                tar_beta = net_out['tar_beta']
                tar_trans = net_out['tar_trans']

                

                T= tar_pose.shape[0]

                rec_pose_np = np.zeros((T, 165))
                rec_pose_np[:, :3+63] = rec_motion[:, :3+63]
                rec_pose_np[:, 66:66+3] = rec_motion[:, 66+90:66+93]
                rec_pose_np[:, 66+9:66+90+9] = rec_motion[:, 66:66+90]
                rec_trans_np = rec_motion[:, 309:309+3]
                rec_exp_np = rec_motion[:, 209:209+100]

                rec_pose_np[:, :66+3] = motion_temporal_filter(rec_pose_np[:, :66+3], sigma=3.5)
                rec_pose_np[:, 66+9:66+90+9] = motion_temporal_filter(rec_pose_np[:, 66+9:66+90+9], sigma=1.0)
                rec_trans_np = motion_temporal_filter(rec_trans_np, sigma=3.5)
                

                tar_pose_np = tar_pose.detach().cpu().numpy()
                tar_exp_np = tar_exps.detach().cpu().numpy().reshape(T, 100)
                tar_trans_np = tar_trans.detach().cpu().numpy().reshape(T, 3)

                
                
                rec_trans_np[:, 1] -= (np.mean(rec_trans_np[:, 1]) - 1.3)


                gt_npz = np.load(self.args.data_path+self.args.pose_rep +"/"+test_seq_list.iloc[its]['id']+".npz", allow_pickle=True)
                prefix_kind = self.gen_cfg.checkpoint.split('/')[-2] + '_' + self.gen_cfg.checkpoint.split('/')[-1]
                np.savez(results_save_path + prefix_kind + "_" + "gt_"+test_seq_list.iloc[its]['id']+'.npz',
                    betas=gt_npz["betas"],
                    poses=tar_pose_np,
                    expressions=tar_exp_np,
                    trans=tar_trans_np,
                    
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30 ,
                )
                np.savez(results_save_path + prefix_kind + "_" + "res_"+test_seq_list.iloc[its]['id']+'.npz',
                    betas=gt_npz["betas"],
                    poses=rec_pose_np,
                    expressions=rec_exp_np,
                    trans=rec_trans_np,
                    
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30,
                )
                total_length += T
                other_tools.render_one_sequence(
                    results_save_path + prefix_kind + "_" + "res_"+test_seq_list.iloc[its]['id']+'.npz', 
                    results_save_path + prefix_kind + "_" + "gt_"+test_seq_list.iloc[its]['id']+'.npz', 
                    results_save_path,
                    self.args.data_path+"wave16k/"+test_seq_list.iloc[its]['id']+".wav",
                    self.args.data_path_1+"smplx_models/", 
                    use_matplotlib = False,
                    args = self.args,
                    )
                
        end_time = time.time() - start_time

def parse_args():
    parser = argparse.ArgumentParser(description='mogen evaluation')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    
    parser.add_argument('--out', help='output animation file')
    parser.add_argument('--beats2_args', help='beats2_args')
    
    parser.add_argument('--launcher',
                        choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--device',
                        choices=['cpu', 'cuda'],
                        default='cuda',
                        help='device used for testing')
    
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

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True
    dataset_name = cfg.data.test.dataset_name
    assert dataset_name in ["beats2"]

    
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


    
    device = args.device
    with open(args.beats2_args, 'r') as file:
        beats2_args = Dict(yaml.safe_load(file))

    trainer = BaseTrainer(beats2_args, device, args)
    trainer.test_demo_stmogen(args.out, model)

if __name__ == '__main__':
    main()
