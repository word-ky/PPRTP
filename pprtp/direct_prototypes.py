"""H06-C direct cosine scoring of count-weighted global class means."""
import torch
import torch.nn.functional as F
from pprtp.oracle import features
from pprtp.paired import procrustes,transform
from pprtp.class_prototypes import class_means


def global_means(p,labels,counts):
    classes=torch.unique(labels,sorted=True)
    bank=torch.stack([(p[labels==c]*counts[labels==c,None].to(p.dtype)).sum(0)/counts[labels==c].sum() for c in classes])
    assert torch.isfinite(bank).all()
    return bank,classes


def cosine_scores(z,bank):
    scores=F.normalize(z,dim=1)@F.normalize(bank,dim=1).T
    assert torch.isfinite(scores).all()
    return scores


def analyze_direct(clients,head,anchors,support,test,tensor_hash,metrics,historical,aligned=True):
    def state():
        return dict(clients=[tensor_hash(c.model.state_dict().values()) for c in clients],
            server=tensor_hash(head.state_dict().values()),
            prototypes=[tensor_hash([c.protos[k] for k in sorted(c.protos)]) for c in clients])
    def gradients():
        return [[None if p.grad is None else tensor_hash([p.grad]) for p in m.parameters()]
            for m in [head]+[c.model for c in clients]]
    before=state();grads=gradients();modes=[[m.training for m in c.model.modules()] for c in clients]
    cpu=torch.get_rng_state().clone();devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    cuda=torch.cuda.get_rng_state_all() if devices else []
    with torch.random.fork_rng(devices=devices):
        aa=[features(c,anchors)[0] for c in clients] if aligned else []
        pp=[];ll=[];nn=[];zz=[];yy=[];owners=[];transforms=[];alignment=[];local=[]
        for i,(c,ds) in enumerate(zip(clients,support)):
            z,y=features(c,ds);raw,labels,counts=class_means(z,y)
            assert labels.tolist()==sorted(c.class_set)
            if aligned:
                t,d=procrustes(aa[i],aa[0],identity=i==0)
                d['transform_hash']=tensor_hash(t);alignment.append(d);transforms.append(t)
                p=transform(raw,t);samples=transform(z,t)
            else: p=raw;samples=z
            for j,label in enumerate(labels.tolist()):
                old=next(v for v in historical['prototypes'] if v['client']==i and v['label']==label)
                assert old['count']==counts[j].item() and old['raw_hash']==tensor_hash([raw[j]])
                if aligned: assert old['prototype_hash']==tensor_hash([p[j]])
                local.append(dict(client=i,label=label,count=counts[j].item(),raw_hash=tensor_hash([raw[j]]),prototype_hash=tensor_hash([p[j]])))
            pp.append(p);ll.append(labels);nn.append(counts);zz.append(samples);yy.append(y);owners.extend([i]*len(labels))
        if aligned: assert alignment==historical['alignment'] and tensor_hash(pp)==historical['prototype_hash']
        p=torch.cat(pp);labels=torch.cat(ll);counts=torch.cat(nn)
        assert tensor_hash(ll)==historical['labels_hash'] and tensor_hash(nn)==historical['counts_hash']
        bank,classes=global_means(p,labels,counts)
        assert classes.tolist()==list(range(10)) # bank row c is explicitly global class c
        norms=bank.norm(dim=1);assert torch.isfinite(norms).all() and (norms>0).all()
        pooled=torch.cat(zz);pooled_labels=torch.cat(yy);receipts=[]
        for j,c in enumerate(classes.tolist()):
            exact=pooled[pooled_labels==c].mean(0);delta=bank[j]-exact
            torch.testing.assert_close(bank[j],exact,atol=1e-6,rtol=1e-5)
            ii=(labels==c).nonzero().flatten().tolist()
            receipts.append(dict(label=c,bank_row=j,owners=[owners[k] for k in ii],counts=counts[labels==c].tolist(),
                total_count=counts[labels==c].sum().item(),hash=tensor_hash([bank[j]]),norm=norms[j].item(),
                hierarchical_max_error=delta.abs().max().item(),hierarchical_fro_error=delta.norm().item()))
        values=[];histograms=[]
        with torch.no_grad():
            for i,c in enumerate(clients):
                z,y=features(c,test)
                if aligned: z=transform(z,transforms[i])
                scores=cosine_scores(z,bank);pred=classes[scores.argmax(1)]
                values.append(metrics(pred.cpu(),y.cpu(),c.class_set))
                seen=torch.zeros_like(y,dtype=torch.bool)
                for label in c.class_set: seen|=y==label
                histograms.append(dict(client=i,overall=torch.bincount(pred,minlength=10).tolist(),
                    seen=torch.bincount(pred[seen],minlength=10).tolist(),missing=torch.bincount(pred[~seen],minlength=10).tolist()))
        assert state()==before and gradients()==grads
    assert modes==[[m.training for m in c.model.modules()] for c in clients]
    assert torch.equal(cpu,torch.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all() if devices else []))
    totals={k:[sum(h[k][j] for h in histograms) for j in range(10)] for k in ('overall','seen','missing')}
    vector_bytes=bank.numel()*bank.element_size()
    return dict(metrics={k:sum(v[k] for v in values)/len(values) for k in ('seen','missing','all','macro')},
        per_client=values,global_prototypes=receipts,global_hash=tensor_hash([bank]),global_labels=classes.tolist(),
        local_prototypes=local,alignment=alignment,prototype_norm_min=norms.min().item(),prototype_norm_max=norms.max().item(),
        prediction_histograms=dict(per_client=histograms,total=totals),predicted_class_count=sum(v>0 for v in totals['overall']),
        communication=dict(semantic_uplink_bytes=p.numel()*p.element_size()+labels.numel()*labels.element_size()+counts.numel()*counts.element_size(),
            anchor_uplink_bytes=len(clients)*len(anchors)*bank.shape[1]*bank.element_size() if aligned else 0,
            global_vectors_downlink_per_client=vector_bytes,global_vectors_downlink_total=vector_bytes*len(clients),
            class_ids_downlink_bytes=0,class_order='fixed ascending 0..9; no separate IDs transmitted',
            learned_head_downlink_per_client=sum(p.numel()*p.element_size() for p in head.parameters()),
            learned_head_downlink_total=len(clients)*sum(p.numel()*p.element_size() for p in head.parameters())),
        state_before=before,state_after=state(),rng_cpu_unchanged=True,rng_cuda_unchanged=True,
        module_modes_unchanged=True,existing_gradients_unchanged=True,anchor_labels_used=False,
        fitting=False,test_used_for_transform=False,cosine_logits_finite=True)
