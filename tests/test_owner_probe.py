import unittest
from types import SimpleNamespace
import torch
from torch.utils.data import TensorDataset
from pprtp.owner_probe import provenance,analyze_owner
from pprtp.run import tensor_hash,metrics


class OwnerProbeTest(unittest.TestCase):
    def test_original_indices_counts_and_exclusion(self):
        sets=[sorted([i,(i+1)%10]) for i in range(10)]
        ds=[TensorDataset(torch.zeros(200,2),torch.tensor([c for c in ss for _ in range(100)])) for ss in sets]
        split=dict(class_sets=sets,train_indices=[list(range(i*200,(i+1)*200)) for i in range(10)])
        receipt=provenance(ds,split,list(range(2000,3000)))
        self.assertEqual(receipt['train_indices'],split['train_indices'])
        self.assertEqual(receipt['class_counts'],[200]*10)
        with self.assertRaises(AssertionError): provenance(ds,split,[0])

    def test_owner_fit_state_and_test_label_isolation(self):
        torch.manual_seed(2)
        model=torch.nn.Module()
        model.base=torch.nn.Sequential(torch.nn.Linear(10,10),torch.nn.BatchNorm1d(10))
        model.head=torch.nn.Linear(10,10)
        c=SimpleNamespace(model=model,device='cpu',class_set=[0,1],protos={0:torch.ones(10)})
        train=TensorDataset(torch.eye(10),torch.arange(10))
        a=analyze_owner([c],model.head,[train],train,tensor_hash,metrics)
        test=TensorDataset(torch.eye(10),torch.arange(10).roll(1))
        b=analyze_owner([c],model.head,[train],test,tensor_hash,metrics)
        self.assertEqual(a['fit'],b['fit'])
        self.assertEqual(a['state_before'],a['state_after'])
        self.assertTrue(model.training and model.base[1].training)
