"""H11-A full-data split: anchors selected before accessing any labels."""
import hashlib,json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torchvision.datasets import CIFAR10,CIFAR100
from pprtp.direct_prototypes import analyze_direct

ANCHOR_SEED=161803
ALLOCATION_SEED=110001

def index_hash(indices):
    return hashlib.sha256(json.dumps(indices,separators=(',',':')).encode()).hexdigest()


def reserve_anchors(size):
    return np.random.default_rng(ANCHOR_SEED).permutation(size)[:256].tolist()


def allocate(labels,class_sets,anchors,num_classes=10):
    labels=np.asarray(labels);reserved=np.zeros(len(labels),dtype=bool);reserved[anchors]=True
    rng=np.random.default_rng(ALLOCATION_SEED);indices=[[] for _ in class_sets];owners={}
    for c in range(num_classes):
        owners[c]=[i for i,cs in enumerate(class_sets) if c in cs]
        available=rng.permutation(np.flatnonzero((labels==c)&~reserved))
        for i,part in zip(owners[c],np.array_split(available,len(owners[c]))):indices[i].extend(part.tolist())
    flat=anchors+sum(indices,[])
    assert len(flat)==len(set(flat))==len(labels) and sorted(flat)==list(range(len(labels)))
    counts=[{c:int((labels[ii]==c).sum()) for c in cs} for ii,cs in zip(indices,class_sets)]
    for c,oo in owners.items():assert max(counts[i][c] for i in oo)-min(counts[i][c] for i in oo)<=1
    return dict(class_sets=class_sets,train_indices=indices,train_index_hashes=[index_hash(ii) for ii in indices],
        class_counts=counts,owners=owners,anchor_indices=anchors,anchor_indices_sha256=index_hash(anchors),
        anchor_rng_seed=ANCHOR_SEED,allocation_rng_seed=ALLOCATION_SEED,anchor_labels_used=False,
        anchor_selection_label_blind=True,train_count=sum(map(len,indices)),anchor_count=len(anchors),
        coverage_count=len(flat),coverage_exact=True,client_overlap=0,anchor_train_overlap=0,official_train_test_separate=True)


def prepare_full(root,seed=0):
    train=CIFAR10(root,train=True,download=True)
    anchors=reserve_anchors(len(train.data)) # No label access until after reservation.
    historical=json.loads(Path(f'research_log/H02A/full/artifacts/experiment/fedgh_seed{seed}/split.json').read_text())
    split=allocate(np.asarray(train.targets),historical['class_sets'],anchors)
    assert split['class_sets']==historical['class_sets'] and len(train.data)==50000
    test=CIFAR10(root,train=False,download=True);assert len(test.data)==10000
    def images(ds,ii):
        x=torch.from_numpy(ds.data[ii].copy()).permute(0,3,1,2).float()/255
        return (x-.5)/.5
    local=[TensorDataset(images(train,ii),torch.tensor(np.asarray(train.targets)[ii],dtype=torch.long)) for ii in split['train_indices']]
    public=TensorDataset(images(train,anchors),torch.zeros(256,dtype=torch.long))
    test_indices=list(range(len(test.data)));split['test_indices']=test_indices;split['test_indices_sha256']=index_hash(test_indices)
    split['test_count']=len(test_indices)
    evaluation=TensorDataset(images(test,test_indices),torch.tensor(test.targets,dtype=torch.long))
    return local,evaluation,split,public


def full_readouts(clients,head,anchors,datasets,test,tensor_hash,metrics,construction_output=None,num_classes=10):
    capture={}
    aligned=analyze_direct(clients,head,anchors,datasets,test,tensor_hash,metrics,None,construction_output=capture,num_classes=num_classes)
    native=analyze_direct(clients,head,anchors,datasets,test,tensor_hash,metrics,None,aligned=False,num_classes=num_classes)
    assert aligned['state_before']==aligned['state_after']==native['state_before']==native['state_after']
    signature=lambda r:[(p['client'],p['label'],p['count'],p['raw_hash']) for p in r['local_prototypes']]
    assert signature(aligned)==signature(native)
    for p in aligned['local_prototypes']:
        assert p['count']==int((datasets[p['client']].tensors[1]==p['label']).sum())
    if construction_output is not None: construction_output.update(capture)
    bank=capture['bank'];nb=torch.nn.functional.normalize(bank,dim=1)
    transforms=capture['transforms'];payload=[sum(t.numel()*t.element_size() for t in tt) for tt in transforms]
    return dict(pprtp_h07=aligned,native_global_prototype_cosine_control=native,
        global_prototype_cosine=(nb@nb.T).cpu().tolist(),global_prototype_norms=bank.norm(dim=1).tolist(),
        communication=dict(**aligned['communication'],naive_affine_downlink_per_client=payload,
            naive_affine_downlink_total=sum(payload),includes_redundant_identity_reference=True),
        forward_examples=dict(anchor_per_client=[len(anchors)]*len(clients),prototype_refresh_per_client=[len(d) for d in datasets],
            pprtp_total=len(anchors)*len(clients)+sum(map(len,datasets)),matched_native_extra_refresh_total=sum(map(len,datasets))),
        same_final_state_exact=True,same_raw_means_counts_exact=True)


OWNERSHIP_SEED=120100

def cifar100_ownership(ownership_seed=OWNERSHIP_SEED):
    order=np.random.default_rng(ownership_seed).permutation(100).tolist()
    class_sets=[[] for _ in range(10)]
    for j,label in enumerate(order):
        class_sets[j%10].append(label);class_sets[(j+1)%10].append(label)
    return [sorted(cs) for cs in class_sets],order


def prepare_cifar100(root,ownership_seed=OWNERSHIP_SEED):
    train=CIFAR100(root,train=True,download=True)
    anchors=reserve_anchors(len(train.data)) # Reserve before accessing labels.
    class_sets,order=cifar100_ownership(ownership_seed)
    split=allocate(np.asarray(train.targets),class_sets,anchors,num_classes=100)
    split.update(dataset='CIFAR100',num_classes=100,ownership_seed=ownership_seed,
        ownership_order=order,ownership_order_sha256=index_hash(order),class_sets_sha256=index_hash(class_sets))
    assert len(train.data)==50000
    test=CIFAR100(root,train=False,download=True);assert len(test.data)==10000
    def images(ds,ii):
        x=torch.from_numpy(ds.data[ii].copy()).permute(0,3,1,2).float()/255
        return (x-.5)/.5
    local=[TensorDataset(images(train,ii),torch.tensor(np.asarray(train.targets)[ii],dtype=torch.long)) for ii in split['train_indices']]
    public=TensorDataset(images(train,anchors),torch.zeros(256,dtype=torch.long))
    test_indices=list(range(len(test.data)));split.update(test_indices=test_indices,test_indices_sha256=index_hash(test_indices),test_count=len(test_indices))
    evaluation=TensorDataset(images(test,test_indices),torch.tensor(test.targets,dtype=torch.long))
    return local,evaluation,split,public
