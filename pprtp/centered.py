"""H09-B translation-free cosine scores using existing affine-map objects."""
import torch
from pprtp.direct_prototypes import cosine_scores


def norm_summary(x):
    n=x.norm(dim=1);assert torch.isfinite(n).all()
    return dict(min=n.min().item(),max=n.max().item(),mean=n.mean().item(),p10=torch.quantile(n,.1).item(),p50=torch.quantile(n,.5).item(),p90=torch.quantile(n,.9).item())


def centered_scores(z,owner_prototypes,owner_labels,bank,t,global_only=False):
    mu,rotation,reference_mu=t
    r=z-mu;owner=owner_prototypes-mu;global_residual=bank-reference_mu
    assert owner.dtype==global_residual.dtype==r.dtype==torch.float32
    assert torch.isfinite(owner).all() and torch.isfinite(global_residual).all()
    assert (owner.norm(dim=1)>0).all() and (global_residual.norm(dim=1)>0).all()
    aligned=r@rotation
    native_owner=cosine_scores(r,owner)
    rotated_owner=cosine_scores(aligned,owner@rotation)
    torch.testing.assert_close(native_owner,rotated_owner,atol=2e-5,rtol=2e-5)
    scores=cosine_scores(aligned,global_residual)
    if not global_only: scores[:,owner_labels]=native_owner
    assert torch.isfinite(scores).all()
    info=dict(owner_rotation_max_abs_error=(native_owner-rotated_owner).abs().max().item(),atol=2e-5,rtol=2e-5,dtype=str(r.dtype),
        norms={name:norm_summary(x) for name,x in [('feature_raw',z),('feature_centered',r),('owner_raw',owner_prototypes),('owner_centered',owner),('global_raw',bank),('global_centered',global_residual)]})
    return scores,info
