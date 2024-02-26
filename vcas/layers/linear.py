import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.cuda.amp import custom_fwd, custom_bwd

from .utils import leverage_k, cal_leverage_var

    
class linear(Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input, weight, bias, scheme, idx):
        ctx.save_for_backward(input, weight, bias)
        ctx.scheme = scheme
        ctx.idx = idx
        return F.linear(input, weight, bias)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        idx = ctx.idx
        scheme = ctx.scheme

        C_in = input.shape[-1]
        C_out = grad_output.shape[-1]

        if not scheme.sample:
            grad_output_flatten = grad_output.reshape(-1, C_out)
            input_flatten = input.reshape(-1, C_in)
            grad_input = (grad_output_flatten @ weight).view(*input.shape)
            grad_weight = grad_output_flatten.t() @ input_flatten
            if scheme.cal_var:
                grad_output_flatten = grad_output.reshape(-1, C_out)
                input_flatten = input.reshape(-1, C_in)
                grad_weight_sample = grad_weight.flatten().index_select(0, scheme.sample_perm_list[idx])
                
                scheme.sgd_grad_dict[idx] = grad_weight_sample
                scheme.sgd_grad_mean_dict[idx] = scheme.sgd_grad_mean_dict.get(idx, 0) + grad_weight_sample
                scheme.sgd_sq_grad_mean_dict[idx] = scheme.sgd_sq_grad_mean_dict.get(idx, 0) + grad_weight_sample ** 2
        else:
            grad_output_nonzero = grad_output.index_select(0, scheme.nonzero_data_index)
            input_nonzero = input.index_select(0, scheme.nonzero_data_index)
            grad_output_nonzero_flatten = grad_output_nonzero.reshape(-1, C_out)
            input_nonzero_flatten = input_nonzero.reshape(-1, C_in)

            grad_input = torch.zeros_like(input, dtype=grad_output.dtype)
            grad_input[scheme.nonzero_data_index] = (grad_output_nonzero_flatten.mm(weight)).view(*input_nonzero.shape)

            if not scheme.cal_var:
                grad_weight = leverage_k(grad_output_nonzero_flatten.t(), input_nonzero_flatten, ratio=scheme.w_ratio_list[idx])
            else:
                grad_weight = grad_output_nonzero_flatten.t() @ input_nonzero_flatten
                grad_weight_sample = grad_weight.flatten().index_select(0, scheme.sample_perm_list[idx])
                leverage_var = cal_leverage_var(grad_output_nonzero_flatten.t(), input_nonzero_flatten, ratio=scheme.w_ratio_list[idx])

                scheme.w_var_dict[idx] = scheme.w_var_dict.get(idx, 0) + leverage_var
                scheme.act_var_dict[idx] = scheme.act_var_dict.get(idx, 0) + (grad_weight_sample - scheme.sgd_grad_dict[idx]).norm().square().item()

        if bias is not None:
            grad_bias = grad_output.sum(tuple(range(grad_output.dim())[:-1]))
        else:
            grad_bias = None

        return grad_input, grad_weight, grad_bias, None, None

class Linear(nn.Linear):
    idx = 0

    def __init__(self, linear_ori: nn.Linear, scheme):
        super(Linear, self).__init__(linear_ori.in_features, linear_ori.out_features, linear_ori.bias is not None, linear_ori.weight.device, linear_ori.weight.dtype)
        delattr(self, 'weight')
        delattr(self, 'bias')
        setattr(self, 'weight', linear_ori.weight)
        setattr(self, 'bias', linear_ori.bias)
        
        self.scheme = scheme
        self.scheme.register_linear(self)
        self.idx = Linear.idx
        Linear.idx += 1


    def forward(self, input):
        return linear.apply(input, self.weight, self.bias, self.scheme, self.idx)
    
