import os
import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import numpy as np
import time
import argparse
import mmcv
import torch
import json
from mmcv import DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import (init_dist, load_checkpoint)
from mogen.models import build_architecture
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from mogen.models.transformers.controlnet import ControlT2MHalf
from mogen.models.transformers.controlnet_mcm import ControlT2MHalf_MCM

from mogen.models.builder import build_submodule
from mogen.core.evaluation.utils import calculate_activation_statistics, calculate_frechet_distance, calculate_diversity

def get_train_test_list(datasplit):
    all_list = []
    train_list = []
    for i in range(1,212):
        all_list.append(str(i).zfill(3))

    if datasplit == "cross_genre":
        test_list = ["063", "132", "143", "036", "098", "198", "130", "012", "211", "193", "179", "065", "137", "161", "092", "120", "037", "109", "204", "144"]
        ignor_list = ["116", "117", "118", "119", "120", "121", "122", "123", "202"]+["130"]
    elif datasplit == "cross_dancer":
        test_list = ['001','002','003','004','005','006','007','008','009','010','011','012','013','124','126','128','130','132']
        ignor_list = ['115','117','119','121','122','135','137','139','141','143','145','147'] + ["116", "118", "120", "123", "202", "159"]+["130"]       
    else:
        raise("error of data split!")
    for one in all_list:
        if one not in test_list:
            if one not in ignor_list:
                train_list.append(one)
    return train_list, test_list, ignor_list

def motion_temporal_filter(motion, sigma=1):
    for i in range(motion.shape[1]):
        motion[:, i] = gaussian_filter(motion[:, i],
                                       sigma=sigma,
                                       mode="nearest")
    return motion

def encode_motion(evaluator_model, motion, motion_length, motion_mask, device):
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
            cur_motion_emb = evaluator_model.encode_motion(
                motion=cur_motion,
                motion_length=cur_motion_length,
                motion_mask=cur_motion_mask,
                device=device)
            motion_emb.append(cur_motion_emb.cpu())
            cur_idx += batch_size
    motion_emb = torch.cat(motion_emb, dim=0)
    return motion_emb
def encode_text(evaluator_model, text, token, device):
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
            cur_text_emb = evaluator_model.encode_text(
                text=cur_text, token=cur_token, device=device)
            text_emb.append(cur_text_emb.cpu())
            cur_idx += batch_size
    text_emb = torch.cat(text_emb, dim=0)
    return text_emb

def finedance_eval(model, 
                   evaluator_model, 
                   motion_length,
                   pre_frames=30,
                   repaint=False,
                   overlap_len=30,
                   fix_very_first=True,
                   device="cuda",
                   motion_dir="./data/datasets/finedance/motion_fea163",
                   music_dir="./data/datasets/finedance/music_npy",
                   text_dir="./data/datasets/finedance/label_json",
                   mean_path="./data/datasets/finedance/mean.npy",
                   std_path="./data/datasets/finedance/std.npy"
                   ):
    train_list, test_list, ignor_list = get_train_test_list("cross_genre")
    

    rec_motion_list = {}
    gt_motion_list = {}
    for test_idx, test_file in tqdm(enumerate(test_list)):
        if test_file in ignor_list:
            continue
        
        gt_finedance_motion = np.load(os.path.join(motion_dir, test_file + '.npy'))
        gt_motion_data = np.zeros((gt_finedance_motion.shape[0], 322))
        gt_motion_data[:, :3+63] = gt_finedance_motion[:, 4+3:4+3+66]
        gt_motion_data[:, 66:66+90] = gt_finedance_motion[:, 4+3+66:4+3+66+90]
        gt_motion_data[:, 309:309+3] = gt_finedance_motion[:, 4:4+3]
        gt_motion_data[:, 309+1] = gt_motion_data[:, 309+1] + 1.3
        
        gt_music_data = np.load(os.path.join(music_dir, test_file + '.npy'))
        
        with open(os.path.join(text_dir, test_file + '.json'), 'r') as file:
            json_text = json.load(file)
            gt_text_data = f"A dancer is performing a {json_text['style1']} dance in the {json_text['style2']} style to the rhythm of the {json_text['name']} song."
        
        before_offset = 360
        gt_motion_data = gt_motion_data[before_offset:]
        gt_music_data = gt_music_data[before_offset:]
        min_all_len = min(min(gt_motion_data.shape[0], gt_music_data.shape[0]), 4096)
        gt_motion_data = gt_motion_data[:min_all_len]
        gt_music_data =  gt_music_data[:min_all_len]

        
        
        
        
        min_all_len = gt_motion_data.shape[0]
        
        num_intervals = 1
        input_dim = 322
        latent_all = torch.zeros(num_intervals, min_all_len, input_dim).to(device)
        
        
        roundt = (min_all_len - pre_frames) // (motion_length - pre_frames)
        remain = (min_all_len - pre_frames) % (motion_length - pre_frames)
        round_l = motion_length - pre_frames
        net_out_val_list = []
        out_motions = []
        print(f'n= ', min_all_len)
        print(f'roundt= ', roundt)
        print(f'remain= ', remain)
        for i in tqdm(range(0, roundt)):
            gt_music_data_tmp = torch.tensor(gt_music_data[i*round_l:(i+1)*round_l+pre_frames, :]).to(device)
            motion_mask = torch.ones(num_intervals, motion_length).to(device)
            if i == 0:
                latent_all_tmp = latent_all[:, i*(round_l):(i+1)*(round_l)+pre_frames, :]
            else:
                latent_all_tmp = latent_all[:, i*(round_l):(i+1)*(round_l)+pre_frames, :]
                if repaint == False:
                    latent_all_tmp[:, :pre_frames, :] = latent_last[:, -pre_frames:, :]
                else:
                    pass
            gt_text_data_tmp = gt_text_data

            metas = []
            metas.append({'text': gt_text_data_tmp})
            input = {
                'motion': latent_all_tmp,
                'motion_mask': motion_mask,
                'motion_length': torch.Tensor([motion_length]).long().to(device),
                'num_intervals': num_intervals,
                'motion_metas': metas,
                'c': gt_music_data_tmp.unsqueeze(0),
            }
            if repaint == True:
                inpaint_dict = {}
                gt_motion_tmp = gt_motion_data[i*(round_l):(i+1)*(round_l)+pre_frames, :]

                if overlap_len > 0:
                    inpaint_dict['gt'] = torch.zeros_like(gt_motion_tmp,  device=device)
                    inpaint_dict['outpainting_mask'] = torch.zeros_like(gt_motion_tmp, 
                                                                        dtype=torch.bool,
                                                                        device=device)  
                    if i == 0:
                        if fix_very_first:
                            inpaint_dict['outpainting_mask'][..., :overlap_len, :] = True  
                            inpaint_dict['gt'][:, :overlap_len, ...] = torch.tensor(gt_motion_tmp)[:, :overlap_len, ...]
                        else:
                            pass
                    elif i > 0:
                        inpaint_dict['outpainting_mask'][..., :overlap_len, :] = True  
                        inpaint_dict['gt'][:, :overlap_len, ...] = outputs[:, -overlap_len:, ...]

                input['y'] = inpaint_dict
                
            all_pred_motion = []
            mean = np.load(mean_path)
            std = np.load(std_path)
            with torch.no_grad():
                input['inference_kwargs'] = {}
                output = model(**input)
                for num_interval_idx in range(num_intervals):
                    pred_motion = output[num_interval_idx]['pred_motion'][:motion_length]
                    pred_motion = pred_motion.detach().cpu().numpy()
                    pred_motion = pred_motion * std + mean
                    all_pred_motion.append(pred_motion)
                pred_motion = np.concatenate(all_pred_motion, axis=0)
                outputs = torch.tensor(pred_motion).to(device).unsqueeze(0)

            if i == roundt - 1:
                out_motions.append(pred_motion)
            else:
                out_motions.append(pred_motion[:round_l])

            if i == 0:
                net_out_val_list.append(pred_motion)
            else:
                net_out_val_list.append(pred_motion[pre_frames:, :])


            latent_last = torch.tensor([(pred_motion - mean) / (std + 1e-8)])

        if repaint == True:
            rec_motion = np.concatenate(out_motions, axis=0)
            rec_motion = torch.tensor(rec_motion).to(device)
        else:
            rec_motion = np.concatenate(net_out_val_list, axis=0)
            rec_motion = torch.tensor(rec_motion).to(device)
        rec_motion_list[str(test_idx)] = rec_motion
        gt_motion_list[str(test_idx)] = torch.tensor(gt_motion_data[:rec_motion.shape[0], :]).to(device)

    
    body_pred_motion_emb_list = []
    body_gt_motion_emb_list = []
    hand_pred_motion_emb_list = []
    hand_gt_motion_emb_list = []
    for test_idx, test_file in tqdm(enumerate(test_list)):
        if test_file in ignor_list:
            continue
        rec_motion = rec_motion_list[str(test_idx)].float()
        
        rec_motion = rec_motion[:4096] 
        
        gt_motion_data = gt_motion_list[str(test_idx)].float()
        assert rec_motion.shape[0] == gt_motion_data.shape[0]
        
        with torch.no_grad():
            pred_motion_emb = encode_motion(
                evaluator_model=evaluator_model,
                motion=rec_motion.unsqueeze(0),
                motion_length=torch.tensor([rec_motion.shape[0]]).to(device),
                motion_mask=rec_motion.unsqueeze(0),
                device=device)
            gt_motion_emb = encode_motion(
                evaluator_model=evaluator_model,
                motion=gt_motion_data.unsqueeze(0),
                motion_length=torch.tensor([gt_motion_data.shape[0]]).to(device),
                motion_mask=rec_motion.unsqueeze(0),
                device=device)
        
        body_pred_motion_emb_list.append(pred_motion_emb)
        body_gt_motion_emb_list.append(gt_motion_emb)
        
        rec_motion_hand = torch.zeros((rec_motion.shape[0], 322)).to(device)
        rec_motion_hand[:, 66:66+90] = rec_motion[:, 66:66+90]
        gt_motion_hand = torch.zeros((gt_motion_data.shape[0], 322)).to(device)
        gt_motion_hand[:, 66:66+90] = gt_motion_data[:, 66:66+90]
        with torch.no_grad():
            pred_motion_emb = encode_motion(
                evaluator_model=evaluator_model,
                motion=rec_motion_hand.unsqueeze(0),
                motion_length=torch.tensor([rec_motion_hand.shape[0]]).to(device),
                motion_mask=rec_motion_hand.unsqueeze(0),
                device=device)
            gt_motion_emb = encode_motion(
                evaluator_model=evaluator_model,
                motion=gt_motion_hand.unsqueeze(0),
                motion_length=torch.tensor([gt_motion_hand.shape[0]]).to(device),
                motion_mask=rec_motion_hand.unsqueeze(0),
                device=device)
        
        hand_pred_motion_emb_list.append(pred_motion_emb)
        hand_gt_motion_emb_list.append(gt_motion_emb)
    pred_motion_emb = torch.cat(body_pred_motion_emb_list, dim=0).cpu().detach().numpy()
    gt_motion_emb = torch.cat(body_gt_motion_emb_list, dim=0).cpu().detach().numpy()
    print(f'pred_motion_emb: {pred_motion_emb.shape}; gt_motion_emb: {gt_motion_emb.shape}')

    hand_pred_motion_emb = torch.cat(hand_pred_motion_emb_list, dim=0).cpu().detach().numpy()
    hand_gt_motion_emb = torch.cat(hand_gt_motion_emb_list, dim=0).cpu().detach().numpy()
    print(f'hand_pred_motion_emb: {hand_pred_motion_emb.shape}; hand_gt_motion_emb: {hand_gt_motion_emb.shape}')

    gt_mu, gt_cov = calculate_activation_statistics(
        gt_motion_emb, 1.0)
    pred_mu, pred_cov = calculate_activation_statistics(
        pred_motion_emb, 1.0)
    fid = calculate_frechet_distance(gt_mu, gt_cov, pred_mu, pred_cov)
    print(f"FID(Whole Body) score: {fid}")

    gt_mu, gt_cov = calculate_activation_statistics(
        hand_gt_motion_emb, 1.0)
    pred_mu, pred_cov = calculate_activation_statistics(
        hand_pred_motion_emb, 1.0)
    fid = calculate_frechet_distance(gt_mu, gt_cov, pred_mu, pred_cov)
    print(f"FID (Hands) score: {fid}")

    
    div = calculate_diversity(pred_motion_emb, diversity_times=pred_motion_emb.shape[0] - 1, emb_scale=1.0, norm_scale=1.0)
    print(f"Diversity score: {div}")

def parse_args():
    parser = argparse.ArgumentParser(description='mogen evaluation')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
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

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args
    
def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    
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
    
    finedance_eval(
        model=model,
        evaluator_model=evaluator_model,
        motion_length=120 if cfg.model.model.type == 'STMoGenTransformer' else 196,
    )

            
if __name__ == "__main__":
    main()