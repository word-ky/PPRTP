"""H03-A unlabeled paired-anchor Procrustes diagnostic."""
import hashlib,json
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torchvision.datasets import CIFAR10
from pprtp.oracle import features
from pprtp.fedgh import fit_linear


def select_anchors(size,excluded):
    excluded=set(excluded)
    pool=np.array([i for i in range(size) if i not in excluded])
    return np.random.default_rng(161803).permutation(pool)[:1000].tolist()


def prepare_paired(root,split,oracle_indices,support_indices,anchor_indices=None):
    ds=CIFAR10(root,train=True,download=False)
    assert ds.train
    excluded=set(sum(split['train_indices'],[]))|set(oracle_indices)|set(sum(support_indices,[]))
    indices=select_anchors(len(ds.data),excluded) if anchor_indices is None else anchor_indices
    assert len(set(indices))==1000 and not excluded.intersection(indices)
    def images(ii):
        x=torch.from_numpy(ds.data[ii].copy()).permute(0,3,1,2).float()/255
        return (x-.5)/.5
    # Anchor labels are not accessed, including for the receipt.
    anchors=TensorDataset(images(indices),torch.zeros(1000,dtype=torch.long))
    support=[TensorDataset(images(ii),torch.tensor(np.asarray(ds.targets)[ii],dtype=torch.long)) for ii in support_indices]
    receipt=dict(indices=indices,indices_sha256=hashlib.sha256(json.dumps(indices,separators=(',',':')).encode()).hexdigest(),
        source='official CIFAR10 train=True',rng_seed=161803,label_blind_selection=True,
        anchor_labels_used=False,reference_client=0,train_oracle_support_overlap=0,test_used_for_fitting=False,
        support_indices=support_indices,support_indices_sha256=hashlib.sha256(json.dumps(support_indices,separators=(',',':')).encode()).hexdigest())
    return anchors,support,receipt


def procrustes(a,reference,identity=False):
    # Double precision SVD; mapped features retain the original float32 dtype.
    x=a.detach().double(); y=reference.detach().double()
    mx=x.mean(0); my=y.mean(0); xc=x-mx; yc=y-my
    if identity:
        rotation=torch.eye(x.shape[1],device=x.device,dtype=x.dtype)
    else:
        u,s,vh=torch.linalg.svd(xc.T@yc,full_matrices=False)
        assert all(torch.isfinite(t).all() for t in (u,s,vh))
        rotation=u@vh
    before=(xc-yc).norm(); after=(xc@rotation-yc).norm()
    ortho=(rotation.T@rotation-torch.eye(rotation.shape[0],device=x.device,dtype=x.dtype)).norm()
    transform=(mx.to(a.dtype),rotation.to(a.dtype),my.to(a.dtype))
    assert all(torch.isfinite(t).all() for t in transform)
    info=dict(centered_residual_before=before.item(),centered_residual_after=after.item(),
        raw_residual_before=(x-y).norm().item(),relative_residual_reduction=(1-after/before).item() if before.item() else 0.,
        orthogonality_error_double=ortho.item(),orthogonality_error_applied=(transform[1].T@transform[1]-torch.eye(x.shape[1],device=a.device)).norm().item())
    return transform,info


def transform(z,t):
    mu,r,reference_mu=t
    out=(z-mu)@r+reference_mu
    assert torch.isfinite(out).all()
    return out


def break_pairs(a,client_id):
    permutation=np.random.default_rng(314159+client_id).permutation(len(a))
    perm=torch.tensor(permutation,device=a.device)
    broken=a[perm]
    assert torch.equal(broken[torch.argsort(perm)],a)
    fixed=int((permutation==np.arange(len(a))).sum())
    assert fixed<=.01*len(a)
    values=permutation.tolist()
    return broken,dict(seed=314159+client_id,permutation=values,fixed_points=fixed,
        permutation_sha256=hashlib.sha256(json.dumps(values,separators=(',',':')).encode()).hexdigest(),
        multiset_bitwise_unchanged=True)


def analyze_paired(clients,head,anchors,support,test,tensor_hash,metrics,broken=False,max_iter=100,expected_alignment=None):
    def state():
        return dict(clients=[tensor_hash(c.model.state_dict().values()) for c in clients],
            server=tensor_hash(head.state_dict().values()),
            prototypes=[tensor_hash([c.protos[k] for k in sorted(c.protos)]) for c in clients])
    before=state(); modes=[[m.training for m in c.model.modules()] for c in clients]
    cpu=torch.get_rng_state().clone(); devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    cuda=torch.cuda.get_rng_state_all() if devices else []
    with torch.random.fork_rng(devices=devices):
        anchor_features=[features(c,anchors)[0] for c in clients] # second tensor never used
        transforms=[]; diagnostics=[]; zz=[]; yy=[]; permutations=[]
        for i,c in enumerate(clients):
            a=anchor_features[i]
            if broken and i:
                a,receipt=break_pairs(a,i)
                permutations.append(dict(client=i,**receipt))
            t,d=procrustes(a,anchor_features[0],identity=i==0)
            d['transform_hash']=tensor_hash(t); transforms.append(t); diagnostics.append(d)
            z,y=features(c,support[i]); zz.append(transform(z,t)); yy.append(y)
        if expected_alignment is not None:
            assert diagnostics==expected_alignment, 'H03-A paired alignment mismatch'
        probe,fit=fit_linear(head,torch.cat(zz),torch.cat(yy),zero=True,max_iter=max_iter)
        fit['head_hash']=tensor_hash(probe.state_dict().values())
        values=[]
        for c,t in zip(clients,transforms):
            z,y=features(c,test)
            with torch.no_grad():
                logits=probe(transform(z,t)); assert torch.isfinite(logits).all()
                values.append(metrics(logits.argmax(1).cpu(),y.cpu(),c.class_set))
        assert state()==before
    assert modes==[[m.training for m in c.model.modules()] for c in clients]
    assert torch.equal(cpu,torch.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all() if devices else []))
    result=dict(metrics={k:sum(v[k] for v in values)/len(values) for k in ('seen','missing','all','macro')},
        per_client=values,fit=fit,alignment=diagnostics,state_before=before,state_after=state(),
        rng_cpu_unchanged=True,rng_cuda_unchanged=True,module_modes_unchanged=True,anchor_labels_used=False)
    if broken:
        result['permutations']=permutations
    return result
