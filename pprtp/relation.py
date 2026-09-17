"""H05-A raw centered anchor inner products; no alignment or normalization."""
import hashlib,json
import numpy as np
import torch
from pprtp.oracle import features
from pprtp.fedgh import fit_linear
from pprtp.paired import support_gradient


def relation(z,anchors):
    mu=anchors.mean(0)
    result=(z-mu)@(anchors-mu).T
    assert torch.isfinite(result).all()
    return result


def permutation(client_id,n=256):
    values=list(range(n)) if client_id==0 else np.random.default_rng(314159+client_id).permutation(n).tolist()
    return dict(client=client_id,seed=None if client_id==0 else 314159+client_id,permutation=values,
        fixed_points=sum(j==v for j,v in enumerate(values)),
        permutation_sha256=hashlib.sha256(json.dumps(values,separators=(',',':')).encode()).hexdigest())


def permute_relations(r,values):
    perm=torch.tensor(values,device=r.device)
    out=r[:,perm]
    assert torch.equal(out[:,torch.argsort(perm)],r)
    return out


def analyze_relation(clients,server,anchors,support,test,tensor_hash,metrics,broken=False,conditioned=False,structural_null=False,feature_receipt=False,expected_features=None,expected_casting=None):
    def state():
        return dict(clients=[tensor_hash(c.model.state_dict().values()) for c in clients],
            server=tensor_hash(server.state_dict().values()),
            prototypes=[tensor_hash([c.protos[k] for k in sorted(c.protos)]) for c in clients])
    def gradients():
        return [[None if p.grad is None else tensor_hash([p.grad]) for p in m.parameters()]
                for m in [server]+[c.model for c in clients]]
    before=state(); grads=gradients(); modes=[[m.training for m in c.model.modules()] for c in clients]
    cpu=torch.get_rng_state().clone();devices=list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    cuda=torch.cuda.get_rng_state_all() if devices else []
    with torch.random.fork_rng(devices=devices):
        aa=[features(c,anchors)[0] for c in clients]
        grams=[(a-a.mean(0))@(a-a.mean(0)).T for a in aa]
        assert all(torch.isfinite(g).all() for g in grams)
        gram=[dict(relative_disagreement=((g-grams[0]).norm()/grams[0].norm()).item(),
            hash=tensor_hash([g])) for g in grams]
        zz=[];yy=[];tt=[];ty=[];perms=[]
        for i,(c,a,ds) in enumerate(zip(clients,aa,support)):
            z,y=features(c,ds);tz,tlabels=features(c,test)
            r=relation(z,a);tr=relation(tz,a)
            if broken:
                receipt=permutation(i,len(a));values=receipt['permutation']
                r=permute_relations(r,values);tr=permute_relations(tr,values)
                receipt['support_test_multisets_bitwise_unchanged']=True;perms.append(receipt)
            zz.append(r);yy.append(y);tt.append(tr);ty.append(tlabels)
        if structural_null:
            from pprtp.helmert import basis,null_diagnostics
            from pprtp.conditioning import matrix_diagnostics
            raw=torch.cat(zz);q=basis(raw.shape[1],device=raw.device,dtype=raw.dtype)
            structural=dict(support=null_diagnostics(raw),test=null_diagnostics(torch.cat(tt)),
                raw_support_hash=tensor_hash([raw]),basis_hash=tensor_hash([q]),
                raw_spectrum=matrix_diagnostics(raw,input_aware=True))
            zz=[z@q for z in zz];tt=[z@q for z in tt]
            assert all(torch.isfinite(z).all() for z in zz+tt)
        if conditioned:
            from pprtp.conditioning import statistics,apply_condition,matrix_diagnostics
            pooled=torch.cat(zz)
            mean,std=statistics(pooled)
            zz=[apply_condition(z,mean,std) for z in zz]
            tt=[apply_condition(z,mean,std) for z in tt]
            conditioning=dict(mean=mean.cpu().tolist(),std=std.cpu().tolist(),
                statistics_hash=tensor_hash([mean,std]),raw_support_hash=tensor_hash([pooled]),
                before=matrix_diagnostics(pooled,input_aware=structural_null),after=matrix_diagnostics(torch.cat(zz),input_aware=structural_null),
                labels_used=False,test_used=False,input_dtype=str(zz[0].dtype))
        if feature_receipt or expected_features is not None:
            fixed=dict(support_hash=tensor_hash([torch.cat(zz)]),test_hash=tensor_hash([torch.cat(tt)]),
                support_labels_hash=tensor_hash(yy),test_labels_hash=tensor_hash(ty),dtype=str(zz[0].dtype))
        if expected_features is not None:
            assert fixed==expected_features
            zz=[cast_fixed(z) for z in zz];tt=[cast_fixed(z) for z in tt]
        if expected_features is not None:
            casting=dict(input_dtype='torch.float32',solver_dtype='torch.float64',roundtrip_bitwise_equal=True,
                support_hash=tensor_hash([torch.cat(zz)]),test_hash=tensor_hash([torch.cat(tt)]))
        if expected_casting is not None:
            assert casting==expected_casting
            from pprtp.precondition import precondition,apply_precondition
            t,inverse,preconditioning=precondition(torch.cat(zz))
            preconditioning['transform_hash']=tensor_hash([t])
            preconditioning['inverse_hash']=tensor_hash([inverse])
            support_transformed=[apply_precondition(z,t,inverse) for z in zz]
            test_transformed=[apply_precondition(z,t,inverse) for z in tt]
            zz=[a[0] for a in support_transformed];tt=[a[0] for a in test_transformed]
            preconditioning['support_reconstruction_relative']=[a[1] for a in support_transformed]
            preconditioning['test_reconstruction_relative']=[a[1] for a in test_transformed]
        template=torch.nn.Linear(zz[0].shape[1],10,device=zz[0].device,dtype=zz[0].dtype)
        probe,fit=fit_linear(template,torch.cat(zz),torch.cat(yy),zero=True,max_iter=2000)
        fit['head_hash']=tensor_hash(probe.state_dict().values());fit['final_support']=support_gradient(probe,zz,yy)
        values=[]
        with torch.no_grad():
            for c,z,y in zip(clients,tt,ty):
                logits=probe(z);assert torch.isfinite(logits).all()
                values.append(metrics(logits.argmax(1).cpu(),y.cpu(),c.class_set))
        assert state()==before and gradients()==grads
    assert modes==[[m.training for m in c.model.modules()] for c in clients]
    assert torch.equal(cpu,torch.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all() if devices else []))
    result=dict(metrics={k:sum(v[k] for v in values)/len(values) for k in ('seen','missing','all','macro')},
        per_client=values,fit=fit,gram=gram,state_before=before,state_after=state(),
        rng_cpu_unchanged=True,rng_cuda_unchanged=True,module_modes_unchanged=True,existing_gradients_unchanged=True,
        anchor_labels_used=False)
    if structural_null: result['structural_null']=structural
    if conditioned: result['conditioning']=conditioning
    if broken: result['permutations']=perms
    if feature_receipt or expected_features is not None: result['fixed_features']=fixed
    if expected_features is not None: result['casting']=casting
    if expected_casting is not None: result['preconditioning']=preconditioning
    return result


def cast_fixed(x):
    assert x.dtype==torch.float32
    out=x.to(torch.float64)
    assert torch.equal(out.float(),x)
    return out
