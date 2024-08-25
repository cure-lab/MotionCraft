import copy
import os
import os.path
from typing import Optional, Union

import numpy as np
import torch

from .base_dataset import BaseMotionDataset
from .builder import DATASETS

from .EMAGE_2024.dataloaders.beat_motionx import CustomDataset

import yaml
from addict import Dict
from tqdm import tqdm

@DATASETS.register_module()
class SpeechMotionDataset(BaseMotionDataset):
    """SpeechMotion dataset.

    Args:
        speech_dir (str): Path to the directory containing the speech files.
    """

    def __init__(self,
                 data_prefix: str,
                 pipeline: list,
                 dataset_name: Optional[Union[str, None]] = None,
                 fixed_length: Optional[Union[int, None]] = None,
                 ann_file: Optional[Union[str, None]] = None,
                 motion_dir: Optional[Union[str, None]] = None,
                 text_dir: Optional[Union[str, None]] = None,
                 token_dir: Optional[Union[str, None]] = None,
                 clip_feat_dir: Optional[Union[str, None]] = None,
                 eval_cfg: Optional[Union[dict, None]] = None,
                 test_mode: Optional[bool] = False,
                 siamese_mode: Optional[bool] = False,
                 tcomb_mode: Optional[bool] = False,
                 
                 ann_config: Optional[Union[str, None]] = None,

                 ):


        
        self.ann_config = ann_config

        super(SpeechMotionDataset, self).__init__(data_prefix=data_prefix,
                                                pipeline=pipeline,
                                                dataset_name=dataset_name,
                                                fixed_length=fixed_length,
                                                ann_file=ann_file,
                                                motion_dir=motion_dir,
                                                eval_cfg=eval_cfg,
                                                test_mode=test_mode)

    def load_anno(self, name):
        raise NotImplementedError

    def load_annotations(self):
        """Load annotations from ``ann_file`` to ``data_infos``"""
        
        
        with open(self.ann_config, 'r') as file:
            self.s2g_args = Dict(yaml.safe_load(file))
        
        temp_data_infos = CustomDataset(self.s2g_args, self.ann_file.split('/')[-1].split('.')[0])
        self.data_infos = []
        for i in tqdm(range(len(temp_data_infos))):
            results = {}
            
            results['text'] = [
                [temp_data_infos.lang_model.index2word[int(item)] for item in temp_data_infos[i]['word']]
            ]

            
            unique_list_str = []
            for item in results['text'][0]:
                if item not in unique_list_str and item != '':  
                    unique_list_str.append(item)

            
            results['text'][0] = 'A person is doing a speech, and the speech content is ' + \
                                    ' '.join(unique_list_str)

            results['motion'] = np.zeros((temp_data_infos[i]['pose'].shape[0], 322))
            results['motion'][:, :3+63] = temp_data_infos[i]['pose'][:, :3+63]             
            results['motion'][:, 66:66+90] = temp_data_infos[i]['pose'][:, 66+9:66+90+9]   
            results['motion'][:, 66+90:66+93] = temp_data_infos[i]['pose'][:, 66:66+3]     
            results['motion'][:, 209:209+100] = temp_data_infos[i]['facial']               
            results['motion'][:, 309:309+3] = temp_data_infos[i]['trans']                  

            
            results['c'] =  np.array(temp_data_infos[i]['audio'])
            results['dataset_name'] = self.dataset_name
            self.data_infos.append(results)
        del temp_data_infos

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
    def prepare_evaluation(self, ):
        raise NotImplementedError


