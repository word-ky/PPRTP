"""H06-B local owner-class means, optionally mapped by frozen Procrustes."""
import torch
from pprtp.oracle import features
from pprtp.paired import procrustes,transform,support_gradient
from pprtp.fedgh import fit_linear,weighted_ce


def class_means(z,y):
    labels=torch.unique(y,sorted=True)
    counts=torch.stack([(y==c).sum() for c in labels])
    means=torch.stack([z[y==c].mean(0) for c in labels])
    assert torch.isfinite(means).all()
    return means,labels,counts


def analyze_prototypes(clients,head,anchors,support,test,tensor_hash,metrics,aligned=True,expected_alignment=None):
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
        pp=[];labels=[];counts=[];zz=[];yy=[];transforms=[];alignment=[];receipts=[];payload=[]
        for i,(c,ds) in enumerate(zip(clients,support)):
            z,y=features(c,ds)
            raw,lab,count=class_means(z,y)
            assert lab.tolist()==sorted(c.class_set)
            if aligned:
                t,d=procrustes(aa[i],aa[0],identity=i==0)
                d['transform_hash']=tensor_hash(t);alignment.append(d);transforms.append(t)
                samples=transform(z,t);prototypes=transform(raw,t)
                reference,ref_labels,ref_counts=class_means(samples,y)
                assert torch.equal(lab,ref_labels) and torch.equal(count,ref_counts)
                torch.testing.assert_close(prototypes,reference,atol=1e-6,rtol=1e-5)
            else:
                samples=z;prototypes=raw
            pp.append(prototypes);labels.append(lab);counts.append(count);zz.append(samples);yy.append(y)
            for j,cid in enumerate(lab.tolist()):
                receipt=dict(client=i,label=cid,count=count[j].item(),raw_hash=tensor_hash([raw[j]]),
                    prototype_hash=tensor_hash([prototypes[j]]))
                if aligned:
                    delta=prototypes[j]-reference[j]
                    receipt.update(affine_mean_max_error=delta.abs().max().item(),affine_mean_fro_error=delta.norm().item())
                receipts.append(receipt)
            full_vector=z.numel()*z.element_size();full_label=y.numel()*y.element_size()
            proto_vector=prototypes.numel()*prototypes.element_size()
            proto_label=lab.numel()*lab.element_size();proto_count=count.numel()*count.element_size()
            anchor_bytes=len(anchors)*z.shape[1]*z.element_size() if aligned else 0
            payload.append(dict(client=i,support_vectors=len(z),prototype_vectors=len(raw),
                full_vector_bytes=full_vector,full_label_bytes=full_label,prototype_vector_bytes=proto_vector,
                prototype_label_bytes=proto_label,prototype_count_bytes=proto_count,anchor_bytes=anchor_bytes,
                full_semantic_bytes=full_vector+full_label,prototype_semantic_bytes=proto_vector+proto_label+proto_count,
                full_with_anchors_bytes=full_vector+full_label+anchor_bytes,
                prototype_with_anchors_bytes=proto_vector+proto_label+proto_count+anchor_bytes))
        if aligned and expected_alignment is not None: assert alignment==expected_alignment
        x=torch.cat(pp);y=torch.cat(labels);n=torch.cat(counts)
        probe,fit=fit_linear(head,x,y,zero=True,max_iter=2000,counts=n)
        logits=probe(x);loss=weighted_ce(logits,y,n)
        grad=torch.cat([g.reshape(-1) for g in torch.autograd.grad(loss,tuple(probe.parameters()))])
        assert torch.isfinite(loss) and torch.isfinite(grad).all()
        fit.update(head_hash=tensor_hash(probe.state_dict().values()),grad_inf=grad.abs().max().item(),grad_l2=grad.norm().item(),
            prototype_class_counts=torch.bincount(y,minlength=10).tolist(),
            prototype_class_correct=torch.bincount(y[logits.argmax(1)==y],minlength=10).tolist())
        # Individual support is evaluation-only for this prototype-trained head.
        full_support=support_gradient(probe,zz,yy)
        values=[]
        with torch.no_grad():
            for i,c in enumerate(clients):
                z,y=features(c,test)
                if aligned: z=transform(z,transforms[i])
                logits=probe(z);assert torch.isfinite(logits).all()
                values.append(metrics(logits.argmax(1).cpu(),y.cpu(),c.class_set))
        assert state()==before and gradients()==grads
    assert modes==[[m.training for m in c.model.modules()] for c in clients]
    assert torch.equal(cpu,torch.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(cuda,torch.cuda.get_rng_state_all() if devices else []))
    totals={k:sum(p[k] for p in payload) for k in payload[0] if k!='client'}
    totals.update(vector_compression=totals['support_vectors']/totals['prototype_vectors'],
        semantic_byte_compression=totals['full_semantic_bytes']/totals['prototype_semantic_bytes'],
        combined_byte_compression=totals['full_with_anchors_bytes']/totals['prototype_with_anchors_bytes'])
    return dict(metrics={k:sum(v[k] for v in values)/len(values) for k in ('seen','missing','all','macro')},
        per_client=values,fit=fit,full_support_diagnostic=full_support,prototypes=receipts,
        prototype_count=len(receipts),prototype_dtype=str(x.dtype),label_dtype=str(labels[0].dtype),count_dtype=str(n.dtype),
        prototype_hash=tensor_hash(pp),labels_hash=tensor_hash(labels),counts_hash=tensor_hash(counts),
        alignment=alignment,communication=dict(per_client=payload,totals=totals),state_before=before,state_after=state(),
        rng_cpu_unchanged=True,rng_cuda_unchanged=True,module_modes_unchanged=True,existing_gradients_unchanged=True,
        anchor_labels_used=False,test_used_for_fitting=False)
