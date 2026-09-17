"""H02-E fresh owner support; disjoint raw images, no paired anchors."""
import hashlib,json
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torchvision.datasets import CIFAR10


def assign_indices(labels,split,oracle_indices):
    excluded=set(sum(split['train_indices'],[]))|set(oracle_indices)
    rng=np.random.default_rng(271828)
    assigned=[[] for _ in split['class_sets']]
    for label in range(10):
        pool=rng.permutation([int(i) for i in np.flatnonzero(np.asarray(labels)==label) if int(i) not in excluded])
        owners=[i for i,ss in enumerate(split['class_sets']) if label in ss]
        for j,i in enumerate(owners): assigned[i].extend(pool[j*100:(j+1)*100].tolist())
    flat=sum(assigned,[])
    assert len(flat)==len(set(flat))==2000 and not excluded.intersection(flat)
    assert all(len(ii)==200 for ii in assigned)
    assert np.bincount(np.asarray(labels)[flat],minlength=10).tolist()==[200]*10
    for ii,ss in zip(assigned,split['class_sets']):
        assert sorted(set(np.asarray(labels)[ii].tolist()))==ss
        assert all(sum(np.asarray(labels)[ii]==c)==100 for c in ss)
    return assigned


def prepare_heldout(root,split,oracle_indices):
    ds=CIFAR10(root,train=True,download=False)
    assert ds.train
    indices=assign_indices(ds.targets,split,oracle_indices)
    datasets=[]
    for ii in indices:
        x=torch.from_numpy(ds.data[ii].copy()).permute(0,3,1,2).float()/255
        datasets.append(TensorDataset((x-.5)/.5,torch.tensor(np.asarray(ds.targets)[ii],dtype=torch.long)))
    receipt=dict(indices=indices,indices_sha256=hashlib.sha256(json.dumps(indices,separators=(',',':')).encode()).hexdigest(),
        rng_seed=271828,source='official CIFAR10 train=True',samples_per_client=[200]*10,total=2000,
        class_counts=[200]*10,client_training_overlap=0,oracle_overlap=0,cross_client_overlap=0,test_used_for_fitting=False)
    return datasets,receipt
