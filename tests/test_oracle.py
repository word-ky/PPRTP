import unittest
from types import SimpleNamespace
import numpy as np
import torch
from torch.utils.data import TensorDataset
from pprtp.oracle import calibration_indices,analyze
from pprtp.run import tensor_hash,metrics


class OracleTest(unittest.TestCase):
    def test_balanced_deterministic_disjoint(self):
        labels=np.repeat(np.arange(10),120)
        used=[list(range(c*120,c*120+20)) for c in range(10)]
        a=calibration_indices(labels,used)
        self.assertEqual(a,calibration_indices(labels,used))
        self.assertFalse(set(a)&set(sum(used,[])))
        self.assertEqual(np.bincount(labels[a]).tolist(),[100]*10)

    def test_oracle_state_rng_buffers_and_test_label_isolation(self):
        torch.manual_seed(8)
        model=torch.nn.Module()
        model.base=torch.nn.Sequential(torch.nn.Linear(10,10),torch.nn.BatchNorm1d(10))
        model.head=torch.nn.Linear(10,10)
        client=SimpleNamespace(model=model,device='cpu',class_set=[0,1],protos={0:torch.ones(10)})
        ds=TensorDataset(torch.eye(10),torch.arange(10))
        first=analyze([client],model.head,ds,ds,tensor_hash,metrics)
        other=TensorDataset(torch.eye(10),torch.arange(10).roll(1))
        second=analyze([client],model.head,ds,other,tensor_hash,metrics)
        self.assertEqual(first['individual_fits'],second['individual_fits'])
        self.assertEqual(first['shared_fit'],second['shared_fit'])
        self.assertTrue(model.training and model.base[1].training)
        self.assertEqual(first['state_before'],first['state_after'])
        self.assertGreaterEqual(first['individual_fits'][0]['after']['accuracy'],.95)
