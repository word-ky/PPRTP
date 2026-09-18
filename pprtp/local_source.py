"""H07-A ordinary local dataset provenance and fixed source-shift diagnostic."""
import hashlib,json
import torch
from pprtp.owner_probe import provenance


def index_hash(indices):
    return hashlib.sha256(json.dumps(indices,separators=(',',':')).encode()).hexdigest()


def local_provenance(datasets,split,oracle_indices,heldout_indices,anchor_indices,tensor_hash):
    receipt=provenance(datasets,split,oracle_indices)
    ordinary=set(sum(split['train_indices'],[]))
    assert not ordinary.intersection(sum(heldout_indices,[]))
    assert not ordinary.intersection(anchor_indices)
    assert split['official_train_test_separate']
    receipt.update(per_client_index_sha256=[index_hash(ii) for ii in split['train_indices']],
        dataset_tensor_sha256=[tensor_hash(ds.tensors) for ds in datasets],
        class_sets=split['class_sets'],heldout_overlap=0,anchor_overlap=0,
        official_train_test_separate=True,semantic_source='ordinary local train; final round10 eval-mode feature refresh',
        online_client_protos_used=False,heldout_semantics_used=False,
        refresh_forward_examples_per_client=[len(ds) for ds in datasets],refresh_forward_examples_total=sum(len(ds) for ds in datasets))
    return receipt


def source_shift(new,old):
    cosine=torch.nn.functional.cosine_similarity(new,old,dim=1)
    distance=(new-old).norm(dim=1)
    assert torch.isfinite(cosine).all() and torch.isfinite(distance).all()
    return [dict(label=i,cosine=cosine[i].item(),l2=distance[i].item()) for i in range(len(new))]
