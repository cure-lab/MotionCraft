import copy
import os
import os.path
from typing import Optional, Union

import numpy as np
import torch

from .base_dataset import BaseMotionDataset
from .builder import DATASETS

import json
from tqdm import tqdm

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
        test_list = [item for item in test_list if item not in ignor_list]
        return train_list, test_list, ignor_list

@DATASETS.register_module()
class FinedanceMotionDataset(BaseMotionDataset):
    """TextMotion dataset.

    Args:
        text_dir (str): Path to the directory containing the text files.
    """

    def __init__(self,
                 data_prefix: str,
                 pipeline: list,
                 dataset_name: Optional[Union[str, None]] = None,
                 fixed_length: Optional[Union[int, None]] = None,
                 ann_file: Optional[Union[str, None]] = None,
                 motion_dir: Optional[Union[str, None]] = None,
                 text_dir: Optional[Union[str, None]] = None,
                 clip_feat_dir: Optional[Union[str, None]] = None,
                 eval_cfg: Optional[Union[dict, None]] = None,
                 test_mode: Optional[bool] = False,
                 
                 datasplit: Optional[Union[str, None]] = None,
                 music_dir: Optional[Union[str, None]] = None,
                 ):
        
        self.datasplit = datasplit
        self.music_dir = os.path.join(data_prefix, 'datasets', dataset_name,
                                     music_dir)
        self.text_dir = os.path.join(data_prefix, 'datasets', dataset_name,
                                     text_dir)
        if clip_feat_dir is not None:
            self.clip_feat_dir = os.path.join(data_prefix, 'datasets',
                                              dataset_name, clip_feat_dir)
        else:
            self.clip_feat_dir = None
        super(FinedanceMotionDataset, self).__init__(data_prefix=data_prefix,
                                                pipeline=pipeline,
                                                dataset_name=dataset_name,
                                                fixed_length=fixed_length,
                                                ann_file=ann_file,
                                                motion_dir=motion_dir,
                                                eval_cfg=eval_cfg,
                                                test_mode=test_mode)

    def load_annotations(self):
        """Load annotations from ``ann_file`` to ``data_infos``"""
        mode = self.ann_file.split('/')[-1].split('.')[0]
        train_list, test_list, ignor_list = get_train_test_list(self.datasplit)
        if mode == 'train':
            datalist= train_list
        else:
            datalist = test_list

        self.data_infos = []
        for l_idx, line in tqdm(enumerate(datalist)):
            self.data_infos.append(self.load_anno(line))

    def load_anno(self, name):
        results = {}
        
        motion_path = os.path.join(self.motion_dir, name + '.npy')
        motion_data_ori = np.load(motion_path)
        motion_data = np.zeros((motion_data_ori.shape[0], 322))
        motion_data[:, :3+63] = motion_data_ori[:, 4+3:4+3+66]
        motion_data[:, 66:66+90] = motion_data_ori[:, 4+3+66:4+3+66+90]
        motion_data[:, 309:309+3] = motion_data_ori[:, 4:4+3]
        
        motion_data[:, 309+1] = motion_data[:, 309+1] + 1.3 

        
        music_path = os.path.join(self.music_dir, name + '.npy')
        music_data = np.load(music_path)

        
        before_offset = 360
        motion_data = motion_data[before_offset:]
        music_data = music_data[before_offset:]
        
        min_all_len = min(motion_data.shape[0], music_data.shape[0])
        results['motion'] = motion_data[:min_all_len]
        results['c'] =  music_data[:min_all_len]
        

        text_path = os.path.join(self.text_dir, name + '.json')
        text_data = []
        with open(text_path, 'r') as file:
            json_text = json.load(file)
            text_data.append(f"A dancer is performing a {json_text['style1']} dance in the {json_text['style2']} style to the rhythm of the {json_text['name']} song.")
        results['text'] = text_data

        if self.clip_feat_dir is not None:
            clip_feat_path = os.path.join(self.clip_feat_dir, name + '.npy')
            clip_feat = torch.from_numpy(np.load(clip_feat_path))
            results['clip_feat'] = clip_feat

        results['dataset_name'] = self.dataset_name

        return results

    def prepare_data(self, idx: int):
        """"Prepare raw data for the f'{idx'}-th data."""
        results = copy.deepcopy(self.data_infos[idx])
        text_list = results['text']
        idx = np.random.randint(0, len(text_list))
        results['text'] = text_list[idx]
        if 'clip_feat' in results.keys():
            results['clip_feat'] = results['clip_feat'][idx]
        results['dataset_name'] = self.dataset_name
        results = self.pipeline(results)
        return results