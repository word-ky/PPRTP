"""H02-D: diagnostic on original local owner samples only."""
import hashlib
import json
import torch
from pprtp.oracle import features
from pprtp.fedgh import fit_linear


def provenance(datasets,split,oracle_indices):
    indices=split['train_indices']
    assert len(datasets)==len(indices)==10
    assert all(len(ds)==len(idx)==200 for ds,idx in zip(datasets,indices))
    assert not set(sum(indices,[])).intersection(oracle_indices)
    labels=torch.cat([ds.tensors[1] for ds in datasets])
    assert torch.bincount(labels,minlength=10).tolist()==[200]*10
    for ds,classes in zip(datasets,split['class_sets']):
        assert sorted(ds.tensors[1].unique().tolist())==classes
        assert all((ds.tensors[1]==c).sum().item()==100 for c in classes)
    return dict(source='exact original local training TensorDatasets from prepare',train_indices=indices,
        indices_sha256=hashlib.sha256(json.dumps(indices,separators=(',',':')).encode()).hexdigest(),
        samples_per_client=[len(ds) for ds in datasets],total=2000,class_counts=[200]*10,
        oracle_overlap=0,test_used_for_fitting=False)


def analyze_owner(clients,head,datasets,test,tensor_hash,metrics):
    def state():
        return dict(clients=[tensor_hash(c.model.state_dict().values()) for c in clients],
            server=tensor_hash(head.state_dict().values()),
            prototypes=[tensor_hash([c.protos[k] for k in sorted(c.protos)]) for c in clients])
    before=state()
    modes=[[m.training for m in c.model.modules()] for c in clients]
    cpu=torch.get_rng_state().clone()
    devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    cuda=torch.cuda.get_rng_state_all() if devices else []
    with torch.random.fork_rng(devices=devices):
        pairs=[features(c,ds) for c,ds in zip(clients,datasets)]
        z=torch.cat([z for z,y in pairs]); y=torch.cat([y for z,y in pairs])
        probe,fit=fit_linear(head,z,y,zero=True)
        fit['head_hash']=tensor_hash(probe.state_dict().values())
        values=[]
        for c in clients:
            tz,ty=features(c,test)
            with torch.no_grad():
                logits=probe(tz)
                assert torch.isfinite(logits).all()
                values.append(metrics(logits.argmax(1).cpu(),ty.cpu(),c.class_set))
        assert state()==before
    assert torch.equal(cpu,torch.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all() if devices else []))
    assert modes==[[m.training for m in c.model.modules()] for c in clients]
    return dict(metrics={k:sum(v[k] for v in values)/len(values) for k in ('seen','missing','all','macro')},
        per_client=values,fit=fit,state_before=before,state_after=state(),
        rng_cpu_unchanged=True,rng_cuda_unchanged=True,module_modes_unchanged=True)
