"""H04-B seed-specific disjoint provenance and three frozen probe arms."""
import hashlib,json
from torchvision.datasets import CIFAR10
from pprtp.oracle import calibration_indices
from pprtp.heldout import assign_indices
from pprtp.paired import select_anchors,prepare_paired,analyze_paired
from pprtp.owner_probe import analyze_owner
from pprtp.anchor_count import prefix


def index_hash(values):
    return hashlib.sha256(json.dumps(values,separators=(',',':')).encode()).hexdigest()


def construct_indices(labels,split):
    oracle=calibration_indices(labels,split['train_indices'])
    support=assign_indices(labels,split,oracle)
    train=set(sum(split['train_indices'],[]));owner=set(sum(support,[]))
    anchors=select_anchors(len(labels),train|set(oracle)|owner)
    groups=[train,set(oracle),owner,set(anchors)]
    assert all(not a.intersection(b) for i,a in enumerate(groups) for b in groups[i+1:])
    return dict(oracle_indices=oracle,support_indices=support,anchor_indices=anchors,
        oracle_sha256=index_hash(oracle),support_sha256=index_hash(support),anchor_sha256=index_hash(anchors),
        train_sha256=index_hash(split['train_indices']),disjoint=True,anchor_labels_used=False,
        rng_seeds=dict(oracle=314159,support=271828,anchors=161803))


def prepare_cross_seed(root,split):
    ds=CIFAR10(root,train=True,download=False)
    receipt=construct_indices(ds.targets,split)
    anchors,support,paired=prepare_paired(root,split,receipt['oracle_indices'],receipt['support_indices'],receipt['anchor_indices'])
    assert paired['indices_sha256']==receipt['anchor_sha256']
    assert paired['support_indices_sha256']==receipt['support_sha256']
    return anchors,support,receipt


def analyze_cross_seed(clients,head,anchors,support,test,tensor_hash,metrics,indices):
    native=analyze_owner(clients,head,support,test,tensor_hash,metrics,max_iter=2000,audit=True)
    arms=dict(native_2000=native)
    for n in (1000,256):
        subset,receipt=prefix(anchors,indices,n)
        arm=analyze_paired(clients,head,subset,support,test,tensor_hash,metrics,
            max_iter=2000,audit=True,rank_diagnostics=True)
        assert arm['state_before']==native['state_before']
        arm['anchor_receipt']=receipt
        arms[f'paired_{n}_2000']=arm
        print(f'H04-B N={n} complete; support_fit={arm["fit"]["after"]["accuracy"]}',flush=True)
    return arms
