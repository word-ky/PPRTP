"""H10-A fixed LOO90 native-radius rejector; calibration is train-only."""
import math
from contextlib import contextmanager
import torch
import torch.nn.functional as F
from pprtp.oracle import features
from pprtp.direct_prototypes import cosine_scores
from pprtp.class_prototypes import class_means
from pprtp.paired import transform


@contextmanager
def isolated(clients,head,tensor_hash):
    def state(): return dict(clients=[tensor_hash(c.model.state_dict().values()) for c in clients],server=tensor_hash(head.state_dict().values()),prototypes=[tensor_hash([c.protos[k] for k in sorted(c.protos)]) for c in clients])
    def grads(): return [[None if p.grad is None else tensor_hash([p.grad]) for p in m.parameters()] for m in [head]+[c.model for c in clients]]
    before=state();gb=grads();modes=[[m.training for m in c.model.modules()] for c in clients]
    cpu=torch.get_rng_state().clone();devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    cuda=torch.cuda.get_rng_state_all() if devices else []
    receipt=dict(state_before=before)
    with torch.random.fork_rng(devices=devices),torch.no_grad(): yield receipt
    assert before==state() and gb==grads() and modes==[[m.training for m in c.model.modules()] for c in clients]
    assert torch.equal(cpu,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all() if devices else []))
    receipt.update(state_after=state(),state_rng_modes_gradients_unchanged=True)


def loo_means(z,prototype):
    n=len(z)
    return (n*prototype-z)/(n-1)


def loo_radius(z,prototype):
    alpha=.10;n=len(z);rank=min(n,math.ceil((n+1)*(1-alpha)))
    leaveout=loo_means(z,prototype)
    a=1-(F.normalize(z,dim=1)*F.normalize(leaveout,dim=1)).sum(1)
    q=a.kthvalue(rank).values
    assert torch.isfinite(a).all() and torch.isfinite(q)
    return q,a,rank


def calibrate_radii(clients,head,datasets,construction,tensor_hash):
    radii=[];records=[]
    with isolated(clients,head,tensor_hash) as isolation:
        for i,(c,ds,(raw,labels)) in enumerate(zip(clients,datasets,construction['raw_owners'])):
            z,y=features(c,ds);means,classes,counts=class_means(z,y)
            assert torch.equal(classes,labels) and tensor_hash([means])==tensor_hash([raw])
            assert counts.tolist()==[100,100]
            qq=[];rows=[];scores=cosine_scores(z,raw)
            for j,label in enumerate(labels.tolist()):
                mask=y==label;q,a,rank=loo_radius(z[mask],raw[j]);qq.append(q)
                rows.append(dict(client=i,label=label,n=int(mask.sum()),alpha=.10,rank_1indexed=rank,radius=q.item(),
                    raw_owner_hash=tensor_hash([raw[j]]),loo_nonconformity_hash=tensor_hash([a]),
                    loo_nonconformity=dict(min=a.min().item(),max=a.max().item(),mean=a.mean().item(),p10=torch.quantile(a,.1).item(),p50=torch.quantile(a,.5).item(),p90=torch.quantile(a,.9).item()),
                    loo_accept_rate=(a<=q).float().mean().item(),fullmean_selfclass_train_accept_rate=((1-scores[mask,j])<=q).float().mean().item()))
            q=torch.stack(qq).detach();radii.append(q)
            accepted=((1-scores)<=q[None]).any(1)
            for row in rows: row['any_owned_train_accept_rate']=accepted[y==row['label']].float().mean().item()
            records.extend(rows)
    return radii,dict(radii=records,isolation=isolation,source='ordinary local train only; fixed final-state eval/no-grad',
        refresh_forward_examples=sum(len(ds) for ds in datasets),radius_hashes=[tensor_hash([q]) for q in radii],
        incremental_communication_bytes=0,local_scalar_radii_per_client=2,local_radius_bytes_per_client=[q.numel()*q.element_size() for q in radii],
        formal_conformal_guarantee=False)


def route(z,raw,owner_labels,bank,t,radii):
    owner_scores=cosine_scores(z,raw);eligible=(1-owner_scores)<=radii[None]
    accepted=eligible.any(1)
    owned_prediction=owner_labels[owner_scores.masked_fill(~eligible,-torch.inf).argmax(1)]
    # Only the unmasked owner predictor is used by the post-prediction oracle.
    owner_only=owner_labels[owner_scores.argmax(1)]
    missing=torch.ones(len(bank),dtype=torch.bool,device=z.device);missing[owner_labels]=False
    missing_labels=missing.nonzero().flatten()
    missing_prediction=missing_labels[cosine_scores(transform(z,t),bank)[:,missing_labels].argmax(1)]
    prediction=torch.where(accepted,owned_prediction,missing_prediction)
    return prediction,accepted,owner_only,missing_prediction


def evaluate_router(clients,head,test,construction,radii,tensor_hash,metrics):
    values=[];oracles=[];hist=[];diagnostics=[]
    with isolated(clients,head,tensor_hash) as isolation:
        for i,(c,(raw,labels),t,q) in enumerate(zip(clients,construction['raw_owners'],construction['transforms'],radii)):
            z,y=features(c,test)
            pred,accepted,owner_only,missing_pred=route(z,raw,labels,construction['bank'],t,q)
            # Target labels affect only metrics and the explicitly diagnostic oracle.
            seen=torch.zeros(10,dtype=torch.bool,device=z.device);seen[labels]=True
            true_seen=seen[y]
            assert seen[pred[accepted]].all() and (~seen[pred[~accepted]]).all()
            oracle=torch.where(true_seen,owner_only,missing_pred)
            values.append(metrics(pred.cpu(),y.cpu(),c.class_set));oracles.append(metrics(oracle.cpu(),y.cpu(),c.class_set))
            hist.append({k:torch.bincount(pred[mask],minlength=10).tolist() for k,mask in [('overall',torch.ones_like(true_seen)),('seen',true_seen),('missing',~true_seen)]})
            diagnostics.append(dict(client=i,total=len(y),seen_count=int(true_seen.sum()),missing_count=int((~true_seen).sum()),
                seen_accepted=int((true_seen&accepted).sum()),missing_rejected=int((~true_seen&~accepted).sum()),
                native_count=int(accepted.sum()),native_correct=int(((pred==y)&accepted).sum()),
                missing_branch_count=int((~accepted).sum()),missing_branch_correct=int(((pred==y)&~accepted).sum())))
    def rates(v):
        return dict(true_seen_accept_rate=v['seen_accepted']/v['seen_count'],true_missing_reject_rate=v['missing_rejected']/v['missing_count'],
            false_accept_rate=1-v['missing_rejected']/v['missing_count'],false_reject_rate=1-v['seen_accepted']/v['seen_count'],
            native_branch_usage=v['native_count']/v['total'],missing_branch_usage=v['missing_branch_count']/v['total'],
            native_branch_accuracy=v['native_correct']/v['native_count'] if v['native_count'] else None,
            missing_branch_accuracy=v['missing_branch_correct']/v['missing_branch_count'] if v['missing_branch_count'] else None)
    total={k:sum(v[k] for v in diagnostics) for k in diagnostics[0] if k!='client'}
    hist_total={k:[sum(h[k][j] for h in hist) for j in range(10)] for k in hist[0]}
    return dict(metrics={k:sum(v[k] for v in values)/len(values) for k in ('seen','missing','all','macro')},per_client=values,
        prediction_histograms=dict(per_client=hist,total=hist_total),predicted_class_count=sum(v>0 for v in hist_total['overall']),
        routing_diagnostics=dict(total=dict(**total,**rates(total)),per_client=[dict(**v,**rates(v)) for v in diagnostics]),
        oracle_seen_missing_router=dict(metrics={k:sum(v[k] for v in oracles)/len(oracles) for k in ('seen','missing','all','macro')},per_client=oracles,diagnostic_only=True),
        isolation=isolation,radius_hashes=[tensor_hash([q]) for q in radii],global_hash=tensor_hash([construction['bank']]),
        transform_hashes=[tensor_hash(t) for t in construction['transforms']],incremental_communication_bytes=0)
