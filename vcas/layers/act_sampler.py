import torch
import torch.nn as nn
from torch.autograd.function import Function
from torch.cuda.amp import custom_fwd, custom_bwd

from .utils import soft_topk, find_top_sum


class sampling(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, scheme, idx):
        ctx.idx = idx
        ctx.scheme = scheme
        return input

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output: torch.Tensor):
        idx = ctx.idx
        scheme = ctx.scheme
        if scheme.sample:
            if idx == len(scheme.act_ratio_sched) - 1 or scheme.act_ratio_sched[idx] < scheme.act_ratio_sched[idx + 1]:
                if idx == len(scheme.act_ratio_sched) - 1:
                    scheme.nonzero_data_index = torch.arange(grad_output.shape[0], device=grad_output.device)
                N = grad_output.shape[0]
                prob = grad_output.norm(dim=tuple(range(1, grad_output.dim())), p=2)
                k = int(N * scheme.act_ratio_sched[idx])

                prob, mask = soft_topk(prob, k)
                index = torch.nonzero(mask).squeeze(-1)
                if index.shape[0] == 0:
                    return grad_output, None, None
                scheme.nonzero_data_index = index
                grad_input = grad_output * (mask / prob).view(-1, *([1] * (grad_output.dim() - 1)))
                return grad_input, None, None
            else: # not the last layer and ratio equal to the former layer
                return grad_output, None, None
        else:
            if scheme.cal_var:
                grad_norm = torch.norm(grad_output, dim=tuple(range(1, grad_output.dim())), p=2)
                scheme.act_ratio_up_sched[idx] += find_top_sum(grad_norm, scheme.s - scheme.args.s_update_step)
                scheme.act_ratio_down_sched[idx] += find_top_sum(grad_norm, scheme.s + scheme.args.s_update_step)
            return grad_output, None, None

# this layer is placed after each block of layers to sample the activation gradients
class ActSampler(nn.Module):
    idx = 0

    def __init__(self, scheme):
        super(ActSampler, self).__init__()
        self.scheme = scheme
        self.scheme.register_act_sampler(self)
        self.idx = ActSampler.idx
        ActSampler.idx += 1

    def forward(self, input):
        return sampling.apply(input, self.scheme, self.idx)

# helper class to append ActSampler to the end of each basic block specified by the user (like BertLayer in BertModel)
# not using nn.Sequential to allow for flexible forward arguments
# note that the forward method is specifically designed to match the forward method of BertLayer in transformers (which returns a tuple of outputs), thus may need to be modified for other models
class Sequential(nn.Module):
    def __init__(self, module: nn.Module, act_sampler: ActSampler):
        super(Sequential, self).__init__()
        self.module = module
        self.act_sampler = act_sampler

    def forward(self, input, *args, **kwargs):
        output = self.module(input, *args, **kwargs)
        layer_output = output[0]
        layer_output = self.act_sampler(layer_output)
        if len(output) > 1:
            return (layer_output,) + output[1:]
        else:
            return (layer_output,)