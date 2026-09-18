"""H10-B: shared all-class group decision, native within-owner refinement."""
import torch
from pprtp.direct_prototypes import cosine_scores
from pprtp.paired import transform
from pprtp.oracle import features
from pprtp.router import isolated


def refine(z,raw,owner_labels,bank,t):
    shared=cosine_scores(transform(z,t),bank).argmax(1)
    routed=torch.isin(shared,owner_labels)
    native=owner_labels[cosine_scores(z,raw).argmax(1)]
    return torch.where(routed,native,shared),shared,routed


def evaluate_refine(clients,head,test,construction,tensor_hash,metrics):
    values=[];shared_values=[];hist=[];rows=[];prediction_hashes=[];route_hashes=[]
    with isolated(clients,head,tensor_hash) as isolation:
        for i,(c,(raw,labels),t) in enumerate(zip(clients,construction['raw_owners'],construction['transforms'])):
            z,y=features(c,test)
            pred,shared,routed=refine(z,raw,labels,construction['bank'],t)
            prediction_hashes.append(tensor_hash([pred]));route_hashes.append(tensor_hash([routed]))
            # Labels are first consumed here, after both predictions and routing.
            seen=torch.isin(y,labels);correct=pred==y;before=shared==y
            assert torch.equal(correct[~seen],before[~seen])
            assert torch.equal(pred[~routed],shared[~routed])
            values.append(metrics(pred.cpu(),y.cpu(),c.class_set))
            shared_values.append(metrics(shared.cpu(),y.cpu(),c.class_set))
            hist.append({k:torch.bincount(pred[mask],minlength=10).tolist() for k,mask in [('overall',torch.ones_like(seen)),('seen',seen),('missing',~seen)]})
            counts=dict(correct_to_correct=int((seen&before&correct).sum()),
                correct_to_wrong=int((seen&before&~correct).sum()),
                wrong_owned_to_correct=int((seen&~before&routed&correct).sum()),
                wrong_owned_to_wrong=int((seen&~before&routed&~correct).sum()),
                wrong_missing_unchanged=int((seen&~before&~routed).sum()))
            assert sum(counts.values())==int(seen.sum())
            assert int((seen&correct).sum())-int((seen&before).sum())==counts['wrong_owned_to_correct']-counts['correct_to_wrong']
            rows.append(dict(client=i,total=len(y),seen_count=int(seen.sum()),missing_count=int((~seen).sum()),
                native_count=int(routed.sum()),seen_routed=int((seen&routed).sum()),missing_routed=int((~seen&routed).sum()),
                shared_missing_correct=int((~seen&before).sum()),refine_missing_correct=int((~seen&correct).sum()),
                shared_seen_correct=int((seen&before).sum()),refine_seen_correct=int((seen&correct).sum()),transitions=counts))
    total={k:sum(r[k] for r in rows) for k in rows[0] if k not in ('client','transitions')}
    total['transitions']={k:sum(r['transitions'][k] for r in rows) for k in rows[0]['transitions']}
    def rates(r):
        return dict(native_route_rate=r['native_count']/r['total'],true_seen_owner_route_rate=r['seen_routed']/r['seen_count'],true_missing_owner_route_rate=r['missing_routed']/r['missing_count'])
    hist_total={k:[sum(h[k][j] for h in hist) for j in range(10)] for k in hist[0]}
    return dict(metrics={k:sum(v[k] for v in values)/len(values) for k in ('seen','missing','all','macro')},per_client=values,
        shared_per_client=shared_values,prediction_histograms=dict(per_client=hist,total=hist_total),
        predicted_class_count=sum(n>0 for n in hist_total['overall']),diagnostics=dict(total=dict(**total,**rates(total)),per_client=[dict(**r,**rates(r)) for r in rows]),
        prediction_hashes=prediction_hashes,route_hashes=route_hashes,isolation=isolation,missing_correctness_per_example_exact=True,
        global_hash=tensor_hash([construction['bank']]),transform_hashes=[tensor_hash(t) for t in construction['transforms']],
        owner_raw_hashes=[[tensor_hash([p]) for p in raw] for raw,_ in construction['raw_owners']],
        incremental_communication_bytes=0,incremental_persistent_storage_bytes=0)
