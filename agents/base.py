import torch
import torch.nn as nn
from torch.nn import functional as F
import copy
import numpy as np
from pathlib import Path
import shutil
import os
import utils
import torch.distributed as dist

class Base:
    def __init__(self, agent_config):
        super(Base, self).__init__()
        self.config = agent_config
        self.init_model_ckpt = self.config['pretrained']
        self.oracle = self.config['oracle']
        self.mu = self.config['mu']
        self.prompting = False
        self.sep_dec_prompts = False
        self.prompt_type = None
        self.lora = False  # for LoRa
        self.r = None  # for LoRa
        self.multi = False  # for LoRa
        self.ada_weights = False  # for LoRa
        self.fuse_type = 'max'  # for LoRa
        self.train_fuse_type = 'current'  # for LoRa
        self.train_distill_type = None  # for LoRa
        self.model_task_id = 1e7  # for LoRa
        self.layer_keys = None
        self.args = self.config['global_args']

        # saving/loading results and checkpoints
        self.output_dir = self.config['output_dir']
        self.task_model_dir = os.path.join(self.output_dir, 'task_models')
        if utils.is_main_process(): Path(self.task_model_dir).mkdir(parents=True, exist_ok=True)
        dist.barrier()

        # pre-training model ckpt
        if self.init_model_ckpt is not None and self.init_model_ckpt != 'None':
            pre_check_file = os.path.join(self.task_model_dir, '_pre.pth')
            if utils.is_main_process(): shutil.copyfile(self.init_model_ckpt,pre_check_file)
            dist.barrier()

            # dict of ckpts
            self.model_ckpt_history = {'pretrained':pre_check_file}
            
            # list of ckpts
            self.model_ckpt_list = []
            self.model_ckpt_list.append(pre_check_file)
        else:
            self.model_ckpt_history = {}
            self.model_ckpt_list = []
        self.model_ckpt_load = copy.deepcopy(self.model_ckpt_list)
        
        # other dirs
        self.task_dir_dict = {}
        self.task_config_dict = {}

        # dynamic
        self.tasks = []
        self.task_id = 0
        self.current_task = 'init'

        # memory
        self.coreset = []

    def get_num_tasks(self):
        return len(self.model_ckpt_list) - (1 if 'pretrained' in self.model_ckpt_history else 0)

    def prep_model4task(self, task_num=-1, force=False):
        if (task_num < 0) and (not force):
            self.model_task_id = 1e7
        else:
            self.model_task_id = task_num

    def increment_task(self, task_str, task_config):
        
        # add task
        self.tasks.append(task_str)

        # create task directory
        self.task_dir_dict[task_str] = os.path.join(self.output_dir, task_str)
        if utils.is_main_process(): Path(self.task_dir_dict[task_str]).mkdir(parents=True, exist_ok=True)
        self.task_dir = self.task_dir_dict[task_str]
        
        # add ckpt files
        self.model_ckpt_history[task_str] = os.path.join(self.task_model_dir, task_str+'.pth')
        if not self.oracle: self.model_ckpt_load = copy.deepcopy(self.model_ckpt_list)
        self.model_ckpt_list.append(self.model_ckpt_history[task_str])

        # save task config
        self.task_config_dict[task_str] = task_config

    def finish_task (self):
        self.task_id += 1

    def regularize(self, state_dict):
        return torch.zeros((1,), requires_grad=True).cuda()

    def update_model(self, model):
        pass

class Naive(Base):
    def __init__(self, agent_config):
        super(Naive, self).__init__(agent_config)

    def regularize(self, state_dict):
        return torch.zeros((1,), requires_grad=True).cuda()