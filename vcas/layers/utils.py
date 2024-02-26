import torch

# unbiased topk importance sampling
def soft_topk(x: torch.Tensor, k: int):
    N = x.shape[0]
    if k >= N:
        return torch.ones_like(x), torch.ones_like(x, dtype=torch.bool)
    x_sorted, _ = torch.sort(x, dim=0, descending=False)
    prefix_sum = torch.cumsum(x_sorted, dim=0)
    cmp = torch.arange(k + 1 - N, k + 1, 1, device=x.device) - prefix_sum / (x_sorted + 1e-9)
    m = torch.searchsorted(cmp, 0).clamp(N - k + 1, N)
    prob = torch.empty_like(x)
    rand = torch.rand_like(x)
    denominator = prefix_sum.index_select(0, m - 1) + 1e-9
    prob = x * (k - N + m) / denominator # will overflow 1 but doesn't matter since we will clamp later
    prob.clamp_(min=1e-9, max=1.0)
    mask = rand < prob
    return prob, mask

# leverage score sampling
def leverage_k(A: torch.Tensor, B: torch.Tensor, ratio: float):
    K = A.shape[1]
    k = int(K * ratio)
    if k >= K or k == 0:
        return A @ B
    leverage_score = A.norm(dim=0, p=2) * B.norm(dim=1, p=2)
    prob, mask = soft_topk(leverage_score, k)
    index = torch.nonzero(mask).squeeze(-1)
    return (A[:, index] / prob[index]) @ B[index]

# return leverage variance of current ratio
def cal_leverage_var(A: torch.Tensor, B: torch.Tensor, ratio: float):
    K = A.shape[1]
    k = int(K * ratio)
    tmp = (A ** 2).sum(0) * (B ** 2).sum(1)
    leverage_score = tmp.sqrt()
    prob, mask = soft_topk(leverage_score, k)
    var = ((1 / prob - 1) * tmp).sum().item()
    return var

# get the minimum ratio of elements that sum to target_percentage of the total sum
def find_top_sum(tensor: torch.Tensor, target_percentage: float):
    sorted_values = torch.sort(tensor.view(-1), descending=True).values
    target_sum = target_percentage * tensor.sum()

    cumulative_sum = torch.cumsum(sorted_values, dim=0)
    count = torch.sum(cumulative_sum < target_sum).item() + 1
    return min(count / len(sorted_values), 1.0)
