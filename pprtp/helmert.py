"""Fixed coordinate chart of the zero-sum subspace, independent of data."""
import math
import torch


def basis(n,device='cpu',dtype=torch.float32):
    q=torch.zeros(n,n-1,device=device,dtype=torch.float64)
    for k in range(1,n):
        scale=math.sqrt(k*(k+1))
        q[:k,k-1]=1/scale
        q[k,k-1]=-k/scale
    q=q.to(dtype)
    assert torch.isfinite(q).all()
    return q


def null_diagnostics(x):
    d=x.detach().double();sums=d.sum(1)
    energy=sums.square().sum()/d.shape[1]
    return dict(row_sum_abs_max=sums.abs().max().item(),row_sum_rms=sums.square().mean().sqrt().item(),
        ones_energy_fraction=(energy/d.square().sum()).item())
