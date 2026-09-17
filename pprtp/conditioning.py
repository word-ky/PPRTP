"""H05-B shared diagonal affine conditioning, estimated from support only."""
import torch


def statistics(x):
    mean=x.mean(0)
    std=(x-mean).square().mean(0).sqrt()
    assert torch.isfinite(mean).all() and torch.isfinite(std).all() and (std>0).all()
    return mean,std


def apply_condition(x,mean,std):
    out=(x-mean)/std
    assert torch.isfinite(out).all()
    return out


def matrix_diagnostics(x):
    d=x.detach().double();s=torch.linalg.svdvals(d)
    assert torch.isfinite(s).all()
    tolerance=max(d.shape)*torch.finfo(d.dtype).eps*s.max()
    nonzero=s[s>tolerance]
    std=(d-d.mean(0)).square().mean(0).sqrt()
    return dict(std_min=std.min().item(),std_median=std.quantile(.5).item(),std_max=std.max().item(),
        singular_max=s.max().item(),singular_min_nonzero=nonzero.min().item(),rank=len(nonzero),
        tolerance=tolerance.item(),tolerance_rule='max(matrix_shape) * float64_epsilon * largest_singular_value',
        condition_nonzero=(s.max()/nonzero.min()).item(),abs_max=d.abs().max().item(),rms=d.square().mean().sqrt().item(),finite=True)


def raw_head(weight,bias,mean,std):
    raw_weight=weight/std[None,:]
    return raw_weight,bias-raw_weight@mean
