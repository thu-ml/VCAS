from dataclasses import dataclass, field

@dataclass
class VcasSampleArguments:
    act_var_tau: float = field(
        default=0.025,
        metadata={"help": "The acceptable ratio of the activation sampling variance to the SGD variance"},
    )

    w_var_tau: float = field(
        default=0.025,
        metadata={"help": "The acceptable ratio of the weight sampling variance to the SGD variance"},
    )

    cal_var_freq: int = field(
        default=100,
        metadata={"help": "The frequency to calculate variance, eg: 100 means calculate variance every 100 steps"},
    )

    s_update_step: float = field(
        default=0.01,
        metadata={"help": "The step size to update S, which is the gradient norm keep ratio of activation gradients, range (0, 1)"},
    )

    w_ratio_mul: float = field(
        default=0.95,
        metadata={"help": "The multiplier for the update of weight sampling ratio, range (0, 1)"},
    )

    # below are implementation details, change if necessary

    cal_var_m: int = field(
        default=2,
        metadata={"help": "The Monte Carlo repeat times for variance calculation"},
    )

    act_sched_decay: float = field(
        default=0.6,
        metadata={"help": "The EMA decay factor to smooth the update of activation sampling ratio schedule"},
    )

    param_sample_ratio: float = field(
        default=0.01,
        metadata={
            "help": "Parameter-wise sample ratio for variance calculation"
                    "eg: 0.01 with param shape (100, 100) will sample 100*100*0.01=1000 indices to calculate variance"
                    "lower this value to reduce the extra memory usage for variance calculation at the expense of variance estimation accuracy"
        },
    )