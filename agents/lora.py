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
from agents.base import Base

#####################
# Base LoRa class #
#####################
class LoRa(Base):
    def __init__(self, agent_config):
        super(LoRa, self).__init__(agent_config)
        self.lora = True
        self.r = int(self.mu)

        # initialize prompts
        self.freeze_encoders = True
        self.freeze_heads = False
        self.layer_keys = [0]

class MultiLoRa(LoRa):
    def __init__(self, agent_config):
        super(MultiLoRa, self).__init__(agent_config)
        self.freeze_heads = True
        self.multi = True

class AdvTextMultiLoRa(MultiLoRa):
    def __init__(self, agent_config):
        super(AdvTextMultiLoRa, self).__init__(agent_config)
        self.train_distill_type = 'adv_text'
        self.fuse_type = 'last'