import numpy as np
import torch
from torch.utils.data import TensorDataset
from torchvision.datasets import CIFAR10


def partition(labels, seed, clients=10, k=2, per_class=100):
    assert clients >= 10 and 1 <= k <= 10
    rng = np.random.default_rng(seed)
    order = rng.permutation(10)
    sets = [sorted(order[(i + np.arange(k)) % 10].tolist()) for i in range(clients)]
    indices = [[] for _ in sets]
    for c in range(10):
        owners = [i for i, classes in enumerate(sets) if c in classes]
        available = rng.permutation(np.flatnonzero(labels == c))
        assert len(available) >= len(owners)*per_class
        for j, client in enumerate(owners):
            indices[client].extend(available[j*per_class:(j+1)*per_class].tolist())
    return sets, indices


def prepare(root, seed, clients, k, train_per_class, test_per_class):
    train = CIFAR10(root, train=True, download=True)
    test = CIFAR10(root, train=False, download=True)
    labels = np.asarray(train.targets)
    sets, indices = partition(labels, seed, clients, k, train_per_class)
    rng = np.random.default_rng(seed + 10000)
    test_indices = np.concatenate([rng.permutation(np.flatnonzero(np.asarray(test.targets)==c))[:test_per_class]
                                   for c in range(10)])

    def tensors(dataset, idx):
        x = torch.from_numpy(dataset.data[idx].copy()).permute(0,3,1,2).float()/255
        return TensorDataset((x-.5)/.5, torch.tensor(np.asarray(dataset.targets)[idx], dtype=torch.long))

    return [tensors(train, idx) for idx in indices], tensors(test, test_indices), dict(
        class_sets=sets, train_indices=indices, test_indices=test_indices.tolist(),
        official_train_test_separate=True)
