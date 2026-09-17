"""Analysis-only all-class oracle; calibration comes exclusively from train."""
import hashlib
import json
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from torchvision.datasets import CIFAR10
from pprtp.fedgh import fit_linear


def calibration_indices(labels,client_indices):
    used=set(i for indices in client_indices for i in indices)
    rng=np.random.default_rng(314159)
    indices=[]
    for c in range(10):
        pool=[int(i) for i in np.flatnonzero(np.asarray(labels)==c) if int(i) not in used]
        indices.extend(rng.permutation(pool)[:100].tolist())
    assert not used.intersection(indices)
    assert np.array_equal(np.bincount(np.asarray(labels)[indices],minlength=10),np.full(10,100))
    return indices


def calibration(root,split):
    ds=CIFAR10(root,train=True,download=False)
    assert ds.train
    indices=calibration_indices(ds.targets,split['train_indices'])
    x=torch.from_numpy(ds.data[indices].copy()).permute(0,3,1,2).float()/255
    y=torch.tensor(np.asarray(ds.targets)[indices],dtype=torch.long)
    receipt=dict(indices=indices,index_sha256=hashlib.sha256(json.dumps(indices,separators=(',',':')).encode()).hexdigest(),
        source='official CIFAR10 train=True only',rng_seed=314159,class_counts=torch.bincount(y).tolist(),
        disjoint_client_training=True,test_used_for_fitting=False)
    return TensorDataset((x-.5)/.5,y),receipt


@torch.no_grad()
def features(client,ds):
    modes=[(m,m.training) for m in client.model.modules()]
    client.model.eval()
    zz=[]; yy=[]
    for x,y in DataLoader(ds,batch_size=128,shuffle=False):
        z=client.model.base(x.to(client.device))
        assert torch.isfinite(z).all()
        zz.append(z); yy.append(y.to(client.device))
    for module,mode in modes:
        module.training=mode
    return torch.cat(zz),torch.cat(yy)


def analyze(clients,server_head,calibration_data,test_data,tensor_hash,metric_fn):
    def state():
        return dict(clients=[tensor_hash(c.model.state_dict().values()) for c in clients],
            server=tensor_hash(server_head.state_dict().values()),
            prototypes=[tensor_hash([c.protos[k] for k in sorted(c.protos)]) for c in clients])
    before=state()
    cpu_rng=torch.get_rng_state().clone()
    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    individual=[]; shared=[]; fits=[]
    devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        train_features=[]; test_features=[]
        for c in clients:
            z,y=features(c,calibration_data)
            tz,ty=features(c,test_data)
            train_features.append((z,y)); test_features.append((tz,ty))
            head,info=fit_linear(server_head,z,y,zero=True)
            with torch.no_grad():
                logits=head(tz)
                assert torch.isfinite(logits).all()
                individual.append(metric_fn(logits.argmax(1).cpu(),ty.cpu(),c.class_set))
            info['head_hash']=tensor_hash(head.state_dict().values())
            fits.append(info)
            assert state()==before
        head,pooled=fit_linear(server_head,torch.cat([z for z,y in train_features]),
                              torch.cat([y for z,y in train_features]),zero=True)
        for c,(z,y) in zip(clients,test_features):
            with torch.no_grad():
                logits=head(z)
                assert torch.isfinite(logits).all()
                shared.append(metric_fn(logits.argmax(1).cpu(),y.cpu(),c.class_set))
        pooled['head_hash']=tensor_hash(head.state_dict().values())
        assert state()==before
    assert torch.equal(cpu_rng,torch.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all() if devices else []))
    values=dict(oracle_individual=individual,oracle_shared=shared)
    summary={key:{m:float(np.mean([v[m] for v in vv])) for m in ('seen','missing','all','macro')} for key,vv in values.items()}
    return dict(metrics=summary,per_client=values,individual_fits=fits,shared_fit=pooled,
        state_before=before,state_after=state(),rng_cpu_unchanged=True,rng_cuda_unchanged=True)
