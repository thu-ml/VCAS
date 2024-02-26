import torch
import torch.nn as nn
from transformers.trainer import Trainer
from transformers import TrainingArguments, DataCollator, PreTrainedModel, PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction, set_seed
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from typing import Union, Optional, Callable, Dict, List, Tuple, Any
import itertools

from vcas import VcasSampleScheme

class VcasTrainer(Trainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        sample_scheme: VcasSampleScheme = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
    ):
        super(VcasTrainer, self).__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer, model_init, compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)
        self.sample_args = sample_scheme.args
        self.sample_scheme = sample_scheme
        self.completed_steps = 0
        self.flops_ratio = 1.0
        self.flops_ratio_N = 0

    def get_train_dataloader(self) -> DataLoader:
        train_dataloader = super(VcasTrainer, self).get_train_dataloader()
        self.cal_var_data = itertools.cycle(iter(train_dataloader))
        return train_dataloader
    
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        self.completed_steps += 1
        if self.completed_steps % self.sample_args.cal_var_freq == 0:
            scheme = self.sample_scheme
            m = self.sample_args.cal_var_m
            freq = self.sample_args.cal_var_freq

            org_state = torch.get_rng_state().clone()
            scheme.cal_var = True # ready to record gradients and calculate variance
            for i in range(m):
                inputs = next(self.cal_var_data)
                scheme.sample = False # no sampling to record SGD gradients
                set_seed(i)
                super(VcasTrainer, self).training_step(model, inputs)
                self.optimizer.zero_grad()

                scheme.sample = True # activation sampling only to record activation gradients
                for _ in range(m):
                    set_seed(i)
                    super(VcasTrainer, self).training_step(model, inputs)
                    self.optimizer.zero_grad()
            scheme.cal_var = False # switch back to normal sampling mode

            # calculate metrics like FLOPs ratio: keep ratio of total training FLOPs
            act_ratio_sched = scheme.act_ratio_sched
            w_ratio_list = scheme.w_ratio_list

            flops = 3 # 1 forward , 1 activation backward, 1 weight backward
            nb = len(act_ratio_sched) # number of blocks
            nl = len(w_ratio_list) // nb # number of layers per block
            flops_act_only = 1 + 2 * sum(act_ratio_sched) / nb
            flops_act_and_w = 1 + sum(act_ratio_sched) / nb + sum([sum(w_ratio_list[i] for i in range(j * nl, (j + 1) * nl)) / nl * act_ratio_sched[j] for j in range(nb)]) / nb
            flops_ratio_tmp = (flops_act_and_w * freq + flops * m + flops_act_only * m**2) / (flops * freq)

            self.flops_ratio = (self.flops_ratio * self.flops_ratio_N + flops_ratio_tmp) / (self.flops_ratio_N + 1)
            self.flops_ratio_N += 1

            # calculate variance and update the sample scheme accordingly
            scheme.calculate()
            sgd_var, act_var, w_var = scheme.update()
            scheme.reset()
            torch.set_rng_state(org_state)

            # log the metrics
            logs = {
                "steps": self.completed_steps,
                "flops_ratio": self.flops_ratio,
                "flops_ratio_tmp": flops_ratio_tmp,
                "s": scheme.s,
                # "act_ratio_sched": scheme.act_ratio_sched,
                # "w_ratio_list": scheme.w_ratio_list,
                "act_ratio_first": scheme.act_ratio_sched[0],
                "act_ratio_last": scheme.act_ratio_sched[-1],
                "w_ratio[0]": scheme.w_ratio_list[0],
                "sgd_var": sgd_var,
                "act_var": act_var,
                "w_var": w_var,
            }
            self.log(logs)
        return super(VcasTrainer, self).training_step(model, inputs)

            



