"""H06-A change only the unconstrained orthogonal completion."""
import hashlib
import torch


def digest(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def haar(size,seed,device):
    generator=torch.Generator(device='cpu').manual_seed(seed)
    a=torch.randn(size,size,generator=generator,dtype=torch.double)
    q,r=torch.linalg.qr(a)
    q=q*torch.sign(torch.diagonal(r))[None,:]
    return q.to(device)


def complete(x,y,u,s,vh,seed,rank=255):
    d=x.shape[1];tol=d*torch.finfo(torch.double).eps*s.max()
    assert (s[:rank]>tol).all() and (s[rank:]<=tol).all()
    q=haar(d-rank,seed,x.device)
    canonical=u@vh
    rotation=u[:,:rank]@vh[:rank,:]+u[:,rank:]@q@vh[rank:,:]
    identity=torch.eye(d,device=x.device,dtype=torch.double)
    orth=(rotation.T@rotation-identity).norm()
    difference=x@rotation-x@canonical
    old=(x@canonical-y).norm();new=(x@rotation-y).norm()
    assert all(torch.isfinite(v).all() for v in (q,rotation,difference,old,new))
    assert orth<1e-9
    torch.testing.assert_close(x@rotation,x@canonical,atol=1e-9,rtol=1e-9)
    torch.testing.assert_close(old,new,atol=1e-9,rtol=1e-9)
    info=dict(seed=seed,q_hash=digest(q),determinant_sign=torch.linalg.slogdet(q).sign.item(),
        q_orthogonality_error=(q.T@q-torch.eye(len(q),device=x.device,dtype=q.dtype)).norm().item(),
        rotation_orthogonality_error=orth.item(),rank=rank,nullity=d-rank,tolerance=tol.item(),
        singular_values=s.cpu().tolist(),smallest_constrained=s[rank-1].item(),largest_null=s[rank].item(),
        mapped_anchor_max_difference=difference.abs().max().item(),mapped_anchor_fro_difference=difference.norm().item(),
        canonical_residual=old.item(),alternative_residual=new.item(),residual_difference=abs(new.item()-old.item()),
        canonical_rotation_hash=digest(canonical),alternative_rotation_hash=digest(rotation),
        labels_used=False,test_used=False,translation_unchanged=True)
    return rotation,info
