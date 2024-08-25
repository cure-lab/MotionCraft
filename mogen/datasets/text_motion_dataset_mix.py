import copy
from typing import Optional, Union

import numpy as np

from .base_dataset import BaseMotionDataset
from .builder import DATASETS

@DATASETS.register_module()
class TextMixMotionDataset(BaseMotionDataset):
    """TextMixMotion dataset.
        Args:
    """

    def __init__(self,
                 data_prefix: Optional[Union[str, None]] = 'mix',
                 eval_cfg: Optional[Union[dict, None]] = None,
                 test_mode: Optional[bool] = False,):

        super(TextMixMotionDataset, self).__init__(data_prefix=data_prefix,
                                                pipeline=[],
                                                dataset_name='mix',
                                                fixed_length=None,
                                                ann_file='mix',
                                                motion_dir='mix',
                                                eval_cfg=eval_cfg,
                                                test_mode=test_mode)

    def merge_datasets(self, merge_datasets):
        self.data_infos = []
        self.pipelines = {}
        for idx, item_dataset in enumerate(merge_datasets):
            try:
                self.pipelines[item_dataset.dataset.dataset_name] = item_dataset.dataset.pipeline
                self.data_infos += item_dataset.dataset.data_infos * item_dataset.times
            except Exception as e:
                print("No Repeated Dataset Wrapper, should only be used in testing!")
                self.pipelines[item_dataset.dataset_name] = item_dataset.pipeline
                self.data_infos += item_dataset.data_infos


    def load_anno(self, name):
        raise NotImplementedError

    def load_annotations(self):
        pass

    def prepare_data(self, idx: int):
        """"Prepare raw data for the f'{idx'}-th data."""
        results = {}
        results['text'] = copy.deepcopy(self.data_infos[idx]['text'])
        results['motion'] = copy.deepcopy(self.data_infos[idx]['motion'])
        results['dataset_name'] = copy.deepcopy(self.data_infos[idx]['dataset_name'])

        text_list = results['text']
        idx = np.random.randint(0, len(text_list))
        results['text'] = text_list[idx]

        results = self.pipelines[results['dataset_name']](results)
        return results
    
