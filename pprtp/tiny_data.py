"""Canonical raw Tiny-ImageNet; official validation is evaluation-only."""
import hashlib
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import TensorDataset
from torchvision.transforms.functional import to_tensor,normalize
from pprtp.full_data import reserve_anchors,allocate,index_hash

TINY_OWNERSHIP_SEED=120200
RAW_URL='http://cs231n.stanford.edu/tiny-imagenet-200.zip'


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_validation(root,class_to_idx,per_class=50):
    root=Path(root);rows=[line.split() for line in (root/'val/val_annotations.txt').read_text().splitlines()]
    mapping={row[0]:class_to_idx[row[1]] for row in rows}
    files=sorted((root/'val/images').glob('*.JPEG'))
    assert len(mapping)==len(rows)==len(files)
    assert set(mapping)=={p.name for p in files}
    labels=np.array([mapping[p.name] for p in files],dtype=np.int64)
    assert np.array_equal(np.bincount(labels,minlength=len(class_to_idx)),np.full(len(class_to_idx),per_class))
    return files,labels


def index_tiny(root,num_classes=200,train_per_class=500,val_per_class=50):
    root=Path(root)
    train_files=sorted((root/'train').glob('*/images/*.JPEG'))
    anchors=reserve_anchors(len(train_files)) # Only list length enters selection, before labels.
    classes=sorted(p.name for p in (root/'train').iterdir() if p.is_dir())
    assert len(classes)==num_classes and set(classes)==set((root/'wnids.txt').read_text().split())
    class_to_idx={name:i for i,name in enumerate(classes)} # PFLlib ImageFolder lexical mapping.
    labels=np.array([class_to_idx[p.parent.parent.name] for p in train_files],dtype=np.int64)
    assert len(train_files)==num_classes*train_per_class
    assert np.array_equal(np.bincount(labels,minlength=num_classes),np.full(num_classes,train_per_class))
    val_files,val_labels=parse_validation(root,class_to_idx,val_per_class)
    train_names=[p.relative_to(root).as_posix() for p in train_files];val_names=[p.relative_to(root).as_posix() for p in val_files]
    assert not set(train_names)&set(val_names)
    train_hashes=[file_hash(p) for p in train_files];val_hashes=[file_hash(p) for p in val_files]
    manifest=dict(class_to_idx=class_to_idx,class_mapping_sha256=index_hash(class_to_idx),train_files=train_names,val_files=val_names,
        train_files_sha256=index_hash(train_names),val_files_sha256=index_hash(val_names),train_file_hashes=train_hashes,val_file_hashes=val_hashes,
        train_content_manifest_sha256=index_hash(train_hashes),val_content_manifest_sha256=index_hash(val_hashes),
        wnids_sha256=file_hash(root/'wnids.txt'),val_annotations_sha256=file_hash(root/'val/val_annotations.txt'),
        raw_train_class_counts=np.bincount(labels,minlength=num_classes).tolist(),raw_val_class_counts=np.bincount(val_labels,minlength=num_classes).tolist(),
        raw_train_count=len(train_files),raw_val_count=len(val_files),official_test_used=False,validation_used_for_fitting=False,
        index_order='lexically sorted relative raw image paths; ImageFolder class mapping',raw_source_url=RAW_URL)
    return train_files,labels,val_files,val_labels,anchors,manifest


def load_images(paths):
    images=torch.empty((len(paths),3,64,64),dtype=torch.float32)
    for i,path in enumerate(paths):
        with Image.open(path) as raw:
            rgb=raw.convert('RGB');assert rgb.size==(64,64)
            images[i]=normalize(to_tensor(rgb),[.5]*3,[.5]*3)
    return images


def prepare_tiny(root):
    train_files,labels,val_files,val_labels,anchors,manifest=index_tiny(root)
    assert len(anchors)==256
    order=np.random.default_rng(TINY_OWNERSHIP_SEED).permutation(200).tolist()
    sets=[sorted(order[i::10]) for i in range(10)]
    split=allocate(labels,sets,anchors,num_classes=200)
    split.update(manifest,dataset='TinyImageNet',num_classes=200,ownership_seed=TINY_OWNERSHIP_SEED,
        ownership_order=order,ownership_order_sha256=index_hash(order),class_sets_sha256=index_hash(sets),
        test_indices=list(range(len(val_files))),test_indices_sha256=index_hash(list(range(len(val_files)))),test_count=len(val_files),
        evaluation_source='official labeled validation, never official unlabeled test',preprocessing='PIL RGB; ToTensor; Normalize(.5,.5,.5); no augmentation or resize')
    local=[TensorDataset(load_images([train_files[i] for i in ii]),torch.tensor(labels[ii],dtype=torch.long)) for ii in split['train_indices']]
    public=TensorDataset(load_images([train_files[i] for i in anchors]),torch.zeros(256,dtype=torch.long))
    evaluation=TensorDataset(load_images(val_files),torch.tensor(val_labels,dtype=torch.long))
    return local,evaluation,split,public
