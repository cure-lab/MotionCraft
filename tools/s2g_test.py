import os
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
os.environ["DISPLAY"] = ':0.0'
import warnings
warnings.filterwarnings("ignore")
import os.path as osp
import signal
import time
import csv
import sys
import warnings
import random
import torch
import torch.nn as nn
import numpy as np
import time
from loguru import logger
import smplx
import argparse
import mmcv
import torch
import yaml
from addict import Dict
from mmcv import DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (init_dist, load_checkpoint,)
from mogen.models import build_architecture
from mogen.datasets.EMAGE_2024.dataloaders.beat_motionx import CustomDataset
from mogen.datasets.EMAGE_2024.utils import other_tools, metric
from mogen.datasets.EMAGE_2024.dataloaders import data_tools
from mogen.datasets.EMAGE_2024.dataloaders.build_vocab import Vocab
from mogen.datasets.EMAGE_2024.utils import rotation_conversions as rc
from mogen.datasets.EMAGE_2024.models.motion_representation import VAESKConv
import librosa
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from mogen.models.transformers.controlnet import ControlT2MHalf
from mogen.models.transformers.controlnet_mcm import ControlT2MHalf_MCM

from mogen.models.builder import build_submodule
from mogen.core.evaluation.utils import calculate_activation_statistics, calculate_frechet_distance

def motion_temporal_filter(motion, sigma=1):
    for i in range(motion.shape[1]):
        motion[:, i] = gaussian_filter(motion[:, i],
                                       sigma=sigma,
                                       mode="nearest")
    return motion

class BaseTrainer(object):
    def __init__(self, args, device, gen_cfg):
        self.gen_cfg = gen_cfg
        self.args = args   
        self.rank = device     
        self.test_data = CustomDataset(args, "test")
        self.test_loader = torch.utils.data.DataLoader(
            self.test_data, 
            batch_size=1,  
            shuffle=False,  
            num_workers=args.loader_workers,
            drop_last=False,
        )
           
        if args.e_name is not None:
            """
            bugs on DDP training using eval_model, using additional eval_copy for evaluation 
            """
            self.eval_model = VAESKConv(args)
            self.eval_copy = VAESKConv(args).to(self.rank)
            other_tools.load_checkpoints(self.eval_copy, args.data_path+args.e_path, args.e_name)
            other_tools.load_checkpoints(self.eval_model, args.data_path+args.e_path, args.e_name)
            self.eval_model.eval()
            self.eval_copy.eval()


        self.smplx = smplx.create(
            self.args.data_path_1+"smplx_models/", 
            model_type='smplx',
            gender='NEUTRAL_2020', 
            use_face_contour=False,
            num_betas=300,
            num_expression_coeffs=100, 
            ext='npz',
            use_pca=False,
        ).to(self.rank).eval()
        self.avg_vel = np.load(args.data_path+f"weights/mean_vel_{args.pose_rep}.npy")
        self.alignmenter = metric.alignment(0.3, 7, self.avg_vel, upper_body=[3,6,9,12,13,14,15,16,17,18,19,20,21])
        self.align_mask = 60
        self.l1_calculator = metric.L1div()
        
class CustomTrainer(BaseTrainer):
    def __init__(self, args, device, gen_cfg, evaluator_model=None):
        super().__init__(args, device, gen_cfg)
        self.gen_cfg = gen_cfg
        self.args = args   
        self.rank = device   
        self.evaluator_model = evaluator_model

        self.reclatent_loss = nn.MSELoss().to(self.rank)
        self.vel_loss = torch.nn.L1Loss(reduction='mean').to(self.rank)
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

    def _g_test(self, loaded_data, model):
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
        print(f'n= ', n)
        print(f'roundt= ', roundt)
        print(f'remain= ', remain)
        
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
                gt_motion[:, :, 3+63] = tar_pose[i*(round_l):(i+1)*(round_l)+self.args.pre_frames, 3+63]
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
            rec_motion = torch.tensor(rec_motion).to(self.rank)
        else:
            rec_motion = np.concatenate(net_out_val_list, axis=0)
            rec_motion = torch.tensor(rec_motion).to(self.rank)
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
    
    def test(self, results_save_path, model):
        
        start_time = time.time()
        total_length = 0
        test_seq_list = self.test_data.selected_file
        align = 0 
        latent_out = []
        latent_ori = []
        l2_all = 0 
        lvel = 0
        self.smplx.eval()
        self.eval_copy.eval()
        pred_motion_emb_list = []
        gt_motion_emb_list = []
        hand_pred_motion_emb_list = []
        hand_gt_motion_emb_list = []
        with torch.no_grad():
            for its, batch_data in enumerate(self.test_loader):
                loaded_data = self._load_data(batch_data)    
                net_out = self._g_test(loaded_data, model)
                rec_motion = net_out['rec_motion'].float()
                tar_pose = net_out['tar_pose'].float()
                tar_exps = net_out['tar_exps'].float()
                tar_beta = net_out['tar_beta'].float()
                tar_trans = net_out['tar_trans'].float()

                T= tar_pose.shape[0]

                rec_pose = torch.zeros((T, 165)).to(self.rank)
                rec_pose[:, :3+63] = rec_motion[:, :3+63]
                rec_pose[:, 66:66+3] = rec_motion[:, 66+90:66+93]
                rec_pose[:, 66+9:66+90+9] = rec_motion[:, 66:66+90]
                

                rec_trans = rec_motion[:, 309:309+3]
                rec_exp = rec_motion[:, 209:209+100]

                rec_pose = rec_pose.float()
                rec_trans = rec_trans.float()
                rec_exp = rec_exp.float()
                
                B, T, J = 1, tar_pose.shape[0], 55

                
                tar_motion = torch.zeros((T, 322)).to(self.rank)
                tar_motion[:, :3+63] = tar_pose[:, :3+63]             
                tar_motion[:, 66:66+90] = tar_pose[:, 66+9:66+90+9]   
                tar_motion[:, 66+90:66+93] = tar_pose[:, 66:66+3]     
                tar_motion[:, 209:209+100] = tar_exps                 
                tar_motion[:, 309:309+3] = tar_trans                  
                with torch.no_grad():
                    pred_motion_emb = self.encode_motion(
                        motion=rec_motion.unsqueeze(0),
                        motion_length=torch.tensor([rec_motion.shape[0]]).to(self.rank),
                        motion_mask=rec_motion.unsqueeze(0),
                        device=self.rank)
                    gt_motion_emb = self.encode_motion(
                        motion=tar_motion.unsqueeze(0),
                        motion_length=torch.tensor([tar_motion.shape[0]]).to(self.rank),
                        motion_mask=rec_motion.unsqueeze(0),
                        device=self.rank)
                print(f'pred_motion_emb: {pred_motion_emb.shape}; gt_motion_emb: {gt_motion_emb.shape}')
                pred_motion_emb_list.append(pred_motion_emb)
                gt_motion_emb_list.append(gt_motion_emb)

                
                rec_motion_hand = torch.zeros((T, 322)).to(self.rank)
                rec_motion_hand[:, :3+63] = torch.zeros_like(rec_pose[:, :3+63])             
                rec_motion_hand[:, :3] = rec_pose[:, :3]                                     
                rec_motion_hand[:, 66:66+90] = rec_pose[:, 66+9:66+90+9]                     
                rec_motion_hand[:, 66+90:66+93] = torch.zeros_like(rec_pose[:, 66:66+3])     
                rec_motion_hand[:, 209:209+100] = torch.zeros_like(rec_exp)                  
                rec_motion_hand[:, 309:309+3] = rec_trans                  

                tar_motion = torch.zeros((T, 322)).to(self.rank)
                tar_motion[:, :3+63] = torch.zeros_like(tar_pose[:, :3+63])             
                tar_motion[:, :3] = tar_pose[:, :3]                                     
                tar_motion[:, 66:66+90] = tar_pose[:, 66+9:66+90+9]                     
                tar_motion[:, 66+90:66+93] = torch.zeros_like(tar_pose[:, 66:66+3])     
                tar_motion[:, 209:209+100] = torch.zeros_like(tar_exps)                 
                tar_motion[:, 309:309+3] = tar_trans                  
                with torch.no_grad():
                    hand_pred_motion_emb = self.encode_motion(
                        motion=rec_motion_hand.unsqueeze(0),
                        motion_length=torch.tensor([rec_motion_hand.shape[0]]).to(self.rank),
                        motion_mask=rec_motion_hand.unsqueeze(0),
                        device=self.rank)
                    hand_gt_motion_emb = self.encode_motion(
                        motion=tar_motion.unsqueeze(0),
                        motion_length=torch.tensor([tar_motion.shape[0]]).to(self.rank),
                        motion_mask=rec_motion_hand.unsqueeze(0),
                        device=self.rank)
                print(f'hand_pred_motion_emb: {hand_pred_motion_emb.shape}; hand_gt_motion_emb: {hand_gt_motion_emb.shape}')
                hand_pred_motion_emb_list.append(hand_pred_motion_emb)
                hand_gt_motion_emb_list.append(hand_gt_motion_emb)
            

                rec_pose_6d = rc.axis_angle_to_matrix(rec_pose.reshape(B*T, J, 3))
                rec_pose_6d = rc.matrix_to_rotation_6d(rec_pose_6d).reshape(B, T, J*6)
                tar_pose_6d = rc.axis_angle_to_matrix(tar_pose.reshape(B*T, J, 3))
                tar_pose_6d = rc.matrix_to_rotation_6d(tar_pose_6d).reshape(B, T, J*6)

                vertices_rec = self.smplx(
                        betas=tar_beta.reshape(B*T, 300), 
                        transl=rec_trans.reshape(B*T, 3)-rec_trans.reshape(B*T, 3), 
                        expression=tar_exps.reshape(B*T, 100)-tar_exps.reshape(B*T, 100),
                        jaw_pose=rec_pose[:, 66:69], 
                        global_orient=rec_pose[:,:3], 
                        body_pose=rec_pose[:,3:21*3+3], 
                        left_hand_pose=rec_pose[:,25*3:40*3], 
                        right_hand_pose=rec_pose[:,40*3:55*3], 
                        return_joints=True, 
                        leye_pose=rec_pose[:, 69:72], 
                        reye_pose=rec_pose[:, 72:75],
                    )
                vertices_rec_face = self.smplx(
                        betas=tar_beta.reshape(B*T, 300), 
                        transl=rec_trans.reshape(B*T, 3)-rec_trans.reshape(B*T, 3), 
                        expression=rec_exp.reshape(B*T, 100), 
                        jaw_pose=rec_pose[:, 66:69], 
                        global_orient=rec_pose[:,:3]-rec_pose[:,:3], 
                        body_pose=rec_pose[:,3:21*3+3]-rec_pose[:,3:21*3+3],
                        left_hand_pose=rec_pose[:,25*3:40*3]-rec_pose[:,25*3:40*3],
                        right_hand_pose=rec_pose[:,40*3:55*3]-rec_pose[:,40*3:55*3],
                        return_verts=True, 
                        return_joints=True,
                        leye_pose=rec_pose[:, 69:72]-rec_pose[:, 69:72],
                        reye_pose=rec_pose[:, 72:75]-rec_pose[:, 72:75],
                    )
                vertices_tar_face = self.smplx(
                    betas=tar_beta.reshape(B*T, 300), 
                    transl=tar_trans.reshape(B*T, 3)-tar_trans.reshape(B*T, 3), 
                    expression=tar_exps.reshape(B*T, 100), 
                    jaw_pose=tar_pose[:, 66:69], 
                    global_orient=tar_pose[:,:3]-tar_pose[:,:3],
                    body_pose=tar_pose[:,3:21*3+3]-tar_pose[:,3:21*3+3], 
                    left_hand_pose=tar_pose[:,25*3:40*3]-tar_pose[:,25*3:40*3],
                    right_hand_pose=tar_pose[:,40*3:55*3]-tar_pose[:,40*3:55*3],
                    return_verts=True, 
                    return_joints=True,
                    leye_pose=tar_pose[:, 69:72]-tar_pose[:, 69:72],
                    reye_pose=tar_pose[:, 72:75]-tar_pose[:, 72:75],
                )

                joints_rec = vertices_rec["joints"].detach().cpu().numpy().reshape(1, T, 127*3)[0, :T, :55*3]
                facial_rec = vertices_rec_face['vertices'].reshape(1, T, -1)[0, :T]
                facial_tar = vertices_tar_face['vertices'].reshape(1, T, -1)[0, :T]
                face_vel_loss = self.vel_loss(facial_rec[1:, :] - facial_tar[:-1, :], facial_tar[1:, :] - facial_tar[:-1, :])
                l2 = self.reclatent_loss(facial_rec, facial_tar)
                l2_all += l2.item() * T
                lvel += face_vel_loss.item() * T

                _ = self.l1_calculator.run(joints_rec)
                if self.alignmenter is not None:
                    in_audio_eval, sr = librosa.load(self.args.data_path+"wave16k/"+test_seq_list.iloc[its]['id']+".wav")
                    in_audio_eval = librosa.resample(in_audio_eval, orig_sr=sr, target_sr=self.args.audio_sr)
                    a_offset = int(self.align_mask * (self.args.audio_sr / self.args.pose_fps))
                    onset_bt = self.alignmenter.load_audio(in_audio_eval[:int(self.args.audio_sr / self.args.pose_fps*T)], a_offset, len(in_audio_eval)-a_offset, True)
                    beat_vel = self.alignmenter.load_pose(joints_rec, self.align_mask, T-self.align_mask, 30, True)
                    
                    align += (self.alignmenter.calculate_align(onset_bt, beat_vel, 30) * (T-2*self.align_mask))
               
                tar_pose_np = tar_pose.detach().cpu().numpy()
                rec_pose_np = rec_pose.detach().cpu().numpy()
                rec_trans_np = rec_trans.detach().cpu().numpy().reshape(B*T, 3)
                rec_exp_np = rec_exp.detach().cpu().numpy().reshape(B*T, 100) 
                tar_exp_np = tar_exps.detach().cpu().numpy().reshape(B*T, 100)
                tar_trans_np = tar_trans.detach().cpu().numpy().reshape(B*T, 3)
                gt_npz = np.load(self.args.data_path+self.args.pose_rep +"/"+test_seq_list.iloc[its]['id']+".npz", allow_pickle=True)
                np.savez(results_save_path+"gt_"+test_seq_list.iloc[its]['id']+'.npz',
                    betas=gt_npz["betas"],
                    poses=tar_pose_np,
                    expressions=tar_exp_np,
                    trans=tar_trans_np,
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30 ,
                )
                np.savez(results_save_path+"res_"+test_seq_list.iloc[its]['id']+'.npz',
                    betas=gt_npz["betas"],
                    poses=rec_pose_np,
                    expressions=rec_exp_np,
                    trans=rec_trans_np,
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30,
                )
                total_length += T

        logger.info(f"l2 loss: {l2_all/total_length}")
        logger.info(f"lvel loss: {lvel/total_length}")
        
        align_avg = align/(total_length-2*len(self.test_loader)*self.align_mask)
        logger.info(f"align score: {align_avg}")

        l1div = self.l1_calculator.avg()
        logger.info(f"l1div score: {l1div}")

        end_time = time.time() - start_time
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.args.pose_fps)} s motion")

        pred_motion_emb = torch.cat(pred_motion_emb_list, dim=0).cpu().detach().numpy()
        gt_motion_emb = torch.cat(gt_motion_emb_list, dim=0).cpu().detach().numpy()
        print(f'pred_motion_emb: {pred_motion_emb.shape}; gt_motion_emb: {gt_motion_emb.shape}')

        hand_pred_motion_emb = torch.cat(hand_pred_motion_emb_list, dim=0).cpu().detach().numpy()
        hand_gt_motion_emb = torch.cat(hand_gt_motion_emb_list, dim=0).cpu().detach().numpy()
        print(f'hand_pred_motion_emb: {hand_pred_motion_emb.shape}; hand_gt_motion_emb: {hand_gt_motion_emb.shape}')

        gt_mu, gt_cov = calculate_activation_statistics(
            gt_motion_emb, 1.0)
        pred_mu, pred_cov = calculate_activation_statistics(
            pred_motion_emb, 1.0)
        fid = calculate_frechet_distance(gt_mu, gt_cov, pred_mu, pred_cov)
        logger.info(f"FID(Whole Body) score: {fid}")

        gt_mu, gt_cov = calculate_activation_statistics(
            hand_gt_motion_emb, 1.0)
        pred_mu, pred_cov = calculate_activation_statistics(
            hand_pred_motion_emb, 1.0)
        fid = calculate_frechet_distance(gt_mu, gt_cov, pred_mu, pred_cov)
        logger.info(f"FID (Hands) score: {fid}")

    def encode_motion(self, motion, motion_length, motion_mask, device):
        N = motion.shape[0]
        motion_emb = []
        batch_size = 1
        cur_idx = 0
        with torch.no_grad():
            while cur_idx < N:
                cur_motion = motion[cur_idx:cur_idx + batch_size].to(device)
                cur_motion_length = \
                    motion_length[cur_idx: cur_idx + batch_size].to(device)
                cur_motion_mask = \
                    motion_mask[cur_idx: cur_idx + batch_size].to(device)
                cur_motion_emb = self.evaluator_model.encode_motion(
                    motion=cur_motion,
                    motion_length=cur_motion_length,
                    motion_mask=cur_motion_mask,
                    device=device)
                motion_emb.append(cur_motion_emb.cpu())
                cur_idx += batch_size
        motion_emb = torch.cat(motion_emb, dim=0)
        return motion_emb
    def encode_text(self, text, token, device):
        N = len(text)
        text_emb = []
        batch_size = 1
        cur_idx = 0
        with torch.no_grad():
            while cur_idx < N:
                cur_text = text[cur_idx:cur_idx + batch_size]
                if token is None:
                    cur_token = None
                else:
                    cur_token = token[cur_idx:cur_idx + batch_size]
                cur_text_emb = self.evaluator_model.encode_text(
                    text=cur_text, token=cur_token, device=device)
                text_emb.append(cur_text_emb.cpu())
                cur_idx += batch_size
        text_emb = torch.cat(text_emb, dim=0)
        return text_emb
    
def parse_args():
    parser = argparse.ArgumentParser(description='mogen evaluation')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--beats2_args', help='beats2_args')
    parser.add_argument('--gpu_collect',
                        action='store_true',
                        help='whether to use gpu to collect results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
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
    parser.add_argument('--random_seed', type=int, default=42, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args
    
def main():
    args = parse_args()
    other_tools.set_random_seed(args)

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)


    
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

    
    evaluator_model = build_submodule(cfg.data.test.eval_cfg.get('evaluator_model', None))
    if args.device == 'cpu':
        evaluator_model = evaluator_model.cpu()
    else:
        evaluator_model = evaluator_model.to(args.device)
    evaluator_model.eval()

    
    device = args.device
    with open(args.beats2_args, 'r') as file:
        beats2_args = Dict(yaml.safe_load(file))
    trainer = CustomTrainer(beats2_args, device, args, evaluator_model)
    
    trainer.test(args.out, model)
            
if __name__ == "__main__":
    main()