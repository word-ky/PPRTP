"""H08-A test-free lagged aligned-bank construction and separate readout."""
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torchvision.datasets import CIFAR10
from pprtp.oracle import features
from pprtp.paired import procrustes,transform
from pprtp.class_prototypes import class_means
from pprtp.direct_prototypes import global_means,cosine_scores
from pprtp.local_source import local_provenance


def prepare_online(root,datasets,split,tensor_hash):
    oracle=json.loads(Path('research_log/H02C/full/artifacts/experiment/fedgh_seed0/oracle_calibration.json').read_text())['indices']
    support=json.loads(Path('research_log/H02E/full/artifacts/experiment/fedgh_seed0/heldout_owner_support.json').read_text())['indices']
    parent=json.loads(Path('research_log/H03A/gate/artifacts/experiment/fedgh_seed0/paired_anchors.json').read_text())['indices']
    old=json.loads(Path('research_log/H07A/gate/artifacts/experiment/fedgh_seed0/final.json').read_text())['local_source_probe']
    indices=parent[:256]
    assert indices==old['anchor_receipt']['indices']
    source=local_provenance(datasets,split,oracle,support,indices,tensor_hash)
    assert source==old['local_source']
    data=CIFAR10(root,train=True,download=False)
    x=torch.from_numpy(data.data[indices].copy()).permute(0,3,1,2).float()/255
    return TensorDataset((x-.5)/.5,torch.zeros(256,dtype=torch.long)),dict(anchor_receipt=old['anchor_receipt'],local_source=source)


def build_bank(clients,anchors,datasets,tensor_hash):
    # No test data, head fitting, or online client.protos are inputs to construction.
    def state(): return [tensor_hash(c.model.state_dict().values()) for c in clients]
    def grads(): return [[None if p.grad is None else tensor_hash([p.grad]) for p in c.model.parameters()] for c in clients]
    before=state();gb=grads();modes=[[m.training for m in c.model.modules()] for c in clients]
    cpu=torch.get_rng_state().clone();devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    cuda=torch.cuda.get_rng_state_all() if devices else []
    with torch.random.fork_rng(devices=devices),torch.no_grad():
        aa=[features(c,anchors)[0] for c in clients];tt=[];alignment=[];pp=[];ll=[];nn=[];local=[]
        for i,(c,ds) in enumerate(zip(clients,datasets)):
            t,d=procrustes(aa[i],aa[0],identity=i==0);t=tuple(v.detach() for v in t)
            tt.append(t);d['transform_hash']=tensor_hash(t);alignment.append(d)
            z,y=features(c,ds);raw,labels,counts=class_means(z,y)
            assert labels.tolist()==sorted(c.class_set)
            p=transform(raw,t);pp.append(p);ll.append(labels);nn.append(counts)
            for j,label in enumerate(labels.tolist()):
                local.append(dict(client=i,label=label,count=counts[j].item(),raw_hash=tensor_hash([raw[j]]),prototype_hash=tensor_hash([p[j]]),norm=p[j].norm().item()))
        p=torch.cat(pp);labels=torch.cat(ll);counts=torch.cat(nn)
        bank,classes=global_means(p,labels,counts);bank=bank.detach()
        assert classes.tolist()==list(range(10)) and len(local)==20
        norms=bank.norm(dim=1);assert torch.isfinite(norms).all() and (norms>0).all()
        totals=[counts[labels==c].sum().item() for c in range(10)]
        assert totals==[200]*10
        assert not bank.requires_grad and all(not v.requires_grad for t in tt for v in t)
    assert before==state() and gb==grads() and modes==[[m.training for m in c.model.modules()] for c in clients]
    assert torch.equal(cpu,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all() if devices else []))
    info=dict(global_hash=tensor_hash([bank]),alignment=alignment,local_prototypes=local,
        global_prototypes=[dict(label=c,count=totals[c],norm=norms[c].item(),hash=tensor_hash([bank[c]])) for c in range(10)],
        state_before=before,state_after=state(),state_rng_modes_gradients_unchanged=True,detached=True,
        semantic_forward_examples=sum(len(ds) for ds in datasets),anchor_forward_examples=len(clients)*len(anchors),
        semantic_uplink_bytes=p.numel()*p.element_size()+labels.numel()*labels.element_size()+counts.numel()*counts.element_size(),
        anchor_uplink_bytes=sum(a.numel()*a.element_size() for a in aa),
        bank_bytes=bank.numel()*bank.element_size(),bank_downlink_total=len(clients)*bank.numel()*bank.element_size(),
        naive_transform_bytes_per_client=[sum(v.numel()*v.element_size() for v in t) for t in tt])
    return bank,tt,info


def aligned_loss(z,y,bank,t,seen,all_classes=True,scale=10.):
    logits=scale*cosine_scores(transform(z,tuple(v.detach() for v in t)),bank.detach())
    assert torch.isfinite(logits).all()
    if not all_classes:
        mask=torch.zeros(len(bank),dtype=torch.bool,device=z.device);mask[seen]=True
        logits=logits.masked_fill(~mask[None],-torch.inf)
    loss=F.cross_entropy(logits,y);assert torch.isfinite(loss)
    return loss


def gradient_direction(z,y,bank,t,seen,scale):
    ga,gs=[torch.autograd.grad(aligned_loss(z,y,bank,t,seen,a,scale),z,retain_graph=True)[0].flatten() for a in (True,False)]
    probs=(scale*cosine_scores(transform(z.detach(),t),bank)).softmax(1)
    missing=torch.ones(len(bank),dtype=torch.bool,device=z.device);missing[seen]=False
    return dict(cosine=F.cosine_similarity(ga[None],gs[None]).item(),all_norm=ga.norm().item(),seen_norm=gs.norm().item(),missing_probability_mass=probs[:,missing].sum(1).mean().item())


def score_bank(clients,test,bank,tt,metrics):
    values=[];hist=[];devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices),torch.no_grad():
        for c,t in zip(clients,tt):
            z,y=features(c,test);pred=cosine_scores(transform(z,t),bank).argmax(1)
            values.append(metrics(pred.cpu(),y.cpu(),c.class_set))
            seen=torch.zeros_like(y,dtype=torch.bool)
            for label in c.class_set: seen|=y==label
            hist.append({k:torch.bincount(pred[mask],minlength=10).tolist() for k,mask in [('overall',torch.ones_like(seen)),('seen',seen),('missing',~seen)]})
    total={k:[sum(h[k][j] for h in hist) for j in range(10)] for k in hist[0]}
    return dict(metrics={k:sum(v[k] for v in values)/len(values) for k in ('seen','missing','all','macro')},per_client=values,
        prediction_histograms=dict(per_client=hist,total=total),predicted_class_count=sum(v>0 for v in total['overall']))
