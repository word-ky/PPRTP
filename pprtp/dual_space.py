"""Zero-parameter H09-A readout; prediction function has no target labels."""
import torch
from pprtp.direct_prototypes import cosine_scores
from pprtp.paired import transform
from pprtp.oracle import features


def dual_scores(z,owner_prototypes,owner_labels,bank,t):
    scores=cosine_scores(transform(z,t),bank).clone()
    scores[:,owner_labels]=cosine_scores(z,owner_prototypes)
    assert torch.isfinite(scores).all()
    return scores


def distributions(values):
    return {k:dict(mean=v.mean().item(),p10=torch.quantile(v,.1).item(),p50=torch.quantile(v,.5).item(),p90=torch.quantile(v,.9).item()) for k,v in values.items()}


def analyze_dual(clients,head,test,construction,tensor_hash,metrics):
    bank=construction['bank'];tt=construction['transforms'];owners=construction['raw_owners']
    def state():
        return dict(clients=[tensor_hash(c.model.state_dict().values()) for c in clients],server=tensor_hash(head.state_dict().values()),
            prototypes=[tensor_hash([c.protos[k] for k in sorted(c.protos)]) for c in clients])
    def grads(): return [[None if p.grad is None else tensor_hash([p.grad]) for p in m.parameters()] for m in [head]+[c.model for c in clients]]
    before=state();gb=grads();modes=[[m.training for m in c.model.modules()] for c in clients]
    cpu=torch.get_rng_state().clone();devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    cuda=torch.cuda.get_rng_state_all() if devices else []
    per_client=[];hist=[];components=[];score_rows=[];pools={'seen':[],'missing':[]};winners={'seen':[],'missing':[]}
    with torch.random.fork_rng(devices=devices),torch.no_grad():
        for i,(c,t,(raw,labels)) in enumerate(zip(clients,tt,owners)):
            assert labels.tolist()==sorted(c.class_set)
            z,y=features(c,test)
            scores=dual_scores(z,raw,labels,bank,t);pred=scores.argmax(1)
            # All targets enter only below this prediction line, for diagnostics.
            owned=torch.zeros(10,dtype=torch.bool,device=z.device);owned[labels]=True
            missing_labels=(~owned).nonzero().flatten()
            true_seen=owned[y];winning_seen=owned[pred]
            max_seen=scores[:,labels].max(1).values;max_missing=scores[:,missing_labels].max(1).values
            triplet=torch.stack((max_seen,max_missing,max_seen-max_missing),dim=1)
            per_client.append(metrics(pred.cpu(),y.cpu(),c.class_set))
            hist.append({k:torch.bincount(pred[mask],minlength=10).tolist() for k,mask in [('overall',torch.ones_like(true_seen)),('seen',true_seen),('missing',~true_seen)]})
            native=labels[scores[:,labels].argmax(1)];aligned=missing_labels[scores[:,missing_labels].argmax(1)]
            components.append(dict(client=i,native_owner_seen_only_correct=(native[true_seen]==y[true_seen]).sum().item(),seen_count=true_seen.sum().item(),
                aligned_missing_only_correct=(aligned[~true_seen]==y[~true_seen]).sum().item(),missing_count=(~true_seen).sum().item()))
            row={}
            for name,mask in [('seen',true_seen),('missing',~true_seen)]:
                pools[name].append(triplet[mask]);winners[name].append(winning_seen[mask])
                row[name]=dict(scores=distributions(dict(zip(('max_seen_score','max_missing_score','difference'),triplet[mask].T))),native_winner_fraction=winning_seen[mask].float().mean().item())
            score_rows.append(row)
    assert before==state() and gb==grads() and modes==[[m.training for m in c.model.modules()] for c in clients]
    assert torch.equal(cpu,torch.get_rng_state()) and all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all() if devices else []))
    total={k:[sum(h[k][j] for h in hist) for j in range(10)] for k in hist[0]}
    score_summary={name:dict(scores=distributions(dict(zip(('max_seen_score','max_missing_score','difference'),torch.cat(pools[name]).T))),
        native_winner_fraction=torch.cat(winners[name]).float().mean().item(),aligned_winner_fraction=1-torch.cat(winners[name]).float().mean().item()) for name in pools}
    overall=torch.cat([v for group in winners.values() for v in group]).float().mean().item()
    return dict(metrics={k:sum(v[k] for v in per_client)/len(per_client) for k in ('seen','missing','all','macro')},per_client=per_client,
        prediction_histograms=dict(per_client=hist,total=total),predicted_class_count=sum(v>0 for v in total['overall']),
        component_diagnostics=dict(native_owner_seen_only=sum(v['native_owner_seen_only_correct'] for v in components)/sum(v['seen_count'] for v in components),
            aligned_missing_only=sum(v['aligned_missing_only_correct'] for v in components)/sum(v['missing_count'] for v in components),per_client=components),
        score_distributions=dict(pooled=score_summary,per_client=score_rows),winning_group_fraction=dict(native_seen=overall,aligned_missing=1-overall),
        state_before=before,state_after=state(),state_rng_modes_gradients_unchanged=True,scores_finite=True,incremental_communication_bytes=0,
        global_hash=tensor_hash([bank]),transform_hashes=[tensor_hash(t) for t in tt],
        owner_raw_hashes=[[tensor_hash([p]) for p in raw] for raw,_ in owners])
