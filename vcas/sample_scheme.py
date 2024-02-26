import torch

from vcas.layers.act_sampler import ActSampler
from vcas.layers.linear import Linear
from vcas.sample_args import VcasSampleArguments

class VcasSampleScheme:
    def __init__(self, sample_args: VcasSampleArguments):
        self.args = sample_args

        # activation sampling schedule
        self.act_ratio_sched = []
        self.act_ratio_up_sched = []
        self.act_ratio_down_sched = []

        # weight sampling schedule
        self.w_ratio_list = []
        self.w_shape_list = []

        # parameter-wise sample indices for variance calculation
        self.sample_perm_list = []
        
        # temporary store gradients for Monte Carlo estimation, keys: layer idx
        self.sgd_grad_dict = {}
        self.sgd_grad_mean_dict = {}
        self.sgd_sq_grad_mean_dict = {}

        # sgd variance, activation sampling variance, weight sampling variance, keys: layer idx
        self.sgd_var_dict = {}
        self.act_var_dict = {}
        self.w_var_dict = {}

        self.s = 1.0 # the gradient norm keep ratio of activation gradients, determining activation sampling ratio
        
        self.sample = False # whether to sample or not (False to get SGD gradients, True to get sample gradients)
        self.cal_var = False # whether in the variance calculation phase
        
        self.nonzero_data_index = None # the indices of the nonzero data after activation sampling, used and updated across the whole backpropagation process

    def register_linear(self, linear: Linear):
        in_features = linear.in_features
        out_features = linear.out_features
        device = linear.weight.device
        sr = self.args.param_sample_ratio

        self.w_ratio_list.append(1.0)
        self.w_shape_list.append(in_features * out_features)
        self.sample_perm_list.append(torch.randperm(in_features * out_features, device=device) \
            [:int(in_features * out_features * sr)])

    def register_act_sampler(self, act_sampler: ActSampler):
        self.act_ratio_sched.append(1)
        self.act_ratio_up_sched.append(0)
        self.act_ratio_down_sched.append(0)

    def calculate(self):
        m = self.args.cal_var_m
        sr = self.args.param_sample_ratio

        # calculate variance
        for idx in self.sgd_grad_dict.keys():
            self.sgd_var_dict[idx] = torch.sum((self.sgd_sq_grad_mean_dict[idx] / m) - (self.sgd_grad_mean_dict[idx] / m) ** 2).item() * (m / (m - 1)) / sr # unbiasedness correction
            self.act_var_dict[idx] /= m**2 * sr
            self.w_var_dict[idx] /= m**2

        # update activation sampling schedule
        for idx in range(len(self.act_ratio_sched)):
            self.act_ratio_up_sched[idx] /= m
            self.act_ratio_down_sched[idx] /= m

    def update(self):
        sgd_var = sum(self.sgd_var_dict.values())
        act_var = sum(self.act_var_dict.values())
        w_var = sum(self.w_var_dict.values())

        act_ratio_sched_new = None
        tau = self.args.act_var_tau
        step = self.args.s_update_step
        decay = self.args.act_sched_decay

        if act_var / sgd_var > tau: # activation sampling variance is too large, increase the sampling ratio
            self.s = min(1, self.s + step)
            act_ratio_sched_new = self.act_ratio_up_sched
        else: # activation sampling variance is acceptable, decrease the sampling ratio
            self.s = max(0, self.s - step)
            act_ratio_sched_new = self.act_ratio_down_sched

        # assuring the sampling schedule is monotonically increasing
        for idx in range(len(self.act_ratio_sched)):
            for i in range(idx + 1, len(self.act_ratio_sched)):
                act_ratio_sched_new[i] = max(act_ratio_sched_new[i], act_ratio_sched_new[idx])

        # EMA smoothing for the activation sampling schedule update
        for idx in range(len(self.act_ratio_sched)):
            self.act_ratio_sched[idx] = decay * self.act_ratio_sched[idx] + (1 - decay) * act_ratio_sched_new[idx]

        # weight sampling ratio update
        tau = self.args.w_var_tau
        mul = self.args.w_ratio_mul
        for idx in range(len(self.w_ratio_list)):
            if self.w_var_dict[idx] / self.sgd_var_dict[idx] > tau: # weight sampling variance is too large, increase the sampling ratio
                self.w_ratio_list[idx] /= mul
            else: # weight sampling variance is acceptable, decrease the sampling ratio
                self.w_ratio_list[idx] *= mul

        return sgd_var, act_var, w_var

    # reset the temporary storage for Monte Carlo variance estimation
    def reset(self):
        self.sgd_grad_dict = {}
        self.sgd_grad_mean_dict = {}
        self.sgd_sq_grad_mean_dict = {}

        self.sgd_var_dict = {}
        self.act_var_dict = {}
        self.w_var_dict = {}

        self.act_ratio_up_sched = [0] * len(self.act_ratio_up_sched)
        self.act_ratio_down_sched = [0] * len(self.act_ratio_down_sched)

        # re-sample the parameter-wise sample indices for variance calculation
        device = self.sample_perm_list[0].device
        sr = self.args.param_sample_ratio
        for idx in range(len(self.sample_perm_list)):
            N = self.w_shape_list[idx]
            self.sample_perm_list[idx] = torch.randperm(N, device=device)[:int(N * sr)]