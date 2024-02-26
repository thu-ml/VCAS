import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.cuda.amp import custom_fwd, custom_bwd
import numpy as np
import logging

from vcas.layers.linear import Linear
from vcas.layers.act_sampler import ActSampler, Sequential
from vcas.sample_args import VcasSampleArguments
from vcas.sample_scheme import VcasSampleScheme

class VcasModelProcessor:
    # act_sampling_repetend: after which block of layers to add ActSampler, eg: BertLayer
    def __init__(self, model: nn.Module, act_sampling_repetend: nn.Module, scheme: VcasSampleScheme):
        self.model = model
        self.act_sampling_repetend = act_sampling_repetend
        self.args = scheme.args
        self.scheme = scheme

    def process(self):
        self.register_linear_layers_in_repetend(self.model)
        self.register_act_sampling_layers(self.model)

    def register_linear_layers_in_repetend(self, module, in_repetend=False):
        for name, child in module.named_children():
            if isinstance(child, self.act_sampling_repetend):
                self.register_linear_layers_in_repetend(child, True)
            elif isinstance(child, nn.Linear) and in_repetend:
                # replace nn.Linear with Linear
                # print("replacing", module._get_name(), name)
                linear_vcas = Linear(child, self.scheme)
                setattr(module, name, linear_vcas)
            else:
                self.register_linear_layers_in_repetend(child, in_repetend)
        
                

    def register_act_sampling_layers(self, module):
        for name, child in module.named_children():
            if isinstance(child, self.act_sampling_repetend):
                # append ActSampler to act_sampling_repetend
                act_sampler = ActSampler(self.scheme)
                module_new = Sequential(child, act_sampler)
                setattr(module, name, module_new)
            else:
                self.register_act_sampling_layers(child)