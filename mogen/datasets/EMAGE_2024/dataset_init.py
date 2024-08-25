import os
import signal
import time
import csv
import sys
import warnings
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import numpy as np
import time
import pprint
from loguru import logger
import smplx
# from torch.utils.tensorboard import SummaryWriter
# import wandb
import matplotlib.pyplot as plt
from utils import config, logger_tools, other_tools, metric
from dataloaders import data_tools
from dataloaders.build_vocab import Vocab
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func

os.environ["MASTER_ADDR"]='127.0.0.1'
os.environ["MASTER_PORT"]='8675'

args = config.parse_args()
# print(args)

dist.init_process_group(backend="nccl", rank=0, world_size=1)

train_data = __import__(f"dataloaders.{args.dataset}", fromlist=["something"]).CustomDataset(args, "train")
print(len(train_data))
print(train_data[0].keys())
for key in list(train_data[0].keys()):
    try:
        print(key, ': ', train_data[0][key].shape)
    except:
        print(key, ': ', train_data[0][key])
print(train_data[0]['word'])
print(train_data[0]['id'])
print(train_data[0]['emo'])
print(train_data[0]['sem'])
# python mogen/datasets/EMAGE_2024/dataset_init.py --config mogen/datasets/EMAGE_2024/configs/st_mogen_emage.yaml >> mogen/datasets/EMAGE_2024/beat2_dataset_init.log 2>&1