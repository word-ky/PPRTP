"""H05-E full-rank support-only invertible right preconditioner."""
import math
import torch


def precondition(x):
    assert x.dtype==torch.float64
    u,s,vh=torch.linalg.svd(x,full_matrices=False)
    assert len(s)==x.shape[1] and (s>0).all()
    assert all(torch.isfinite(a).all() for a in (u,s,vh))
    scale=math.sqrt(len(x))/s
    t=vh.T*scale
    inverse=(s/math.sqrt(len(x)))[:,None]*vh
    z=x@t
    post=torch.linalg.svdvals(z)
    residual=(z.T@z/len(x)-torch.eye(x.shape[1],device=x.device,dtype=x.dtype)).norm().item()
    assert all(torch.isfinite(a).all() for a in (t,inverse,z,post))
    assert residual<1e-8
    info=dict(singular_before=s.cpu().tolist(),singular_after=post.cpu().tolist(),
        min_s=s[-1].item(),max_s=s[0].item(),condition_before=(s[0]/s[-1]).item(),
        condition_after=(post[0]/post[-1]).item(),max_scaling=scale.max().item(),
        whitening_residual_fro=residual,labels_used=False,test_used=False,dimensions=len(s))
    return t,inverse,info


def apply_precondition(x,t,inverse):
    z=x@t
    error=((z@inverse-x).norm()/x.norm()).item()
    assert torch.isfinite(z).all() and error<1e-9
    return z,error
