import unittest
from types import SimpleNamespace
import torch
from torch.utils.data import TensorDataset
from pprtp.relation import analyze_relation,cast_fixed
from pprtp.run import tensor_hash,metrics


class PrecisionTest(unittest.TestCase):
    def test_fixed_cast_and_probe_identity(self):
        torch.manual_seed(46)
        x=torch.randn(20,255)
        self.assertTrue(torch.equal(cast_fixed(x).float(),x))
        model=torch.nn.Module();model.base=torch.nn.Identity();model.head=torch.nn.Linear(10,10)
        model.train();model.head.eval()
        for p in model.parameters(): p.grad=torch.ones_like(p)
        client=SimpleNamespace(model=model,device='cpu',class_set=[0,1],protos={0:torch.ones(10)})
        anchors=TensorDataset(torch.randn(256,10),torch.zeros(256,dtype=torch.long))
        ds=TensorDataset(torch.eye(10),torch.arange(10))
        args=([client],model.head,anchors,[ds],ds,tensor_hash,metrics)
        a=analyze_relation(*args,conditioned=True,structural_null=True,feature_receipt=True)
        b=analyze_relation(*args,conditioned=True,structural_null=True,expected_features=a['fixed_features'])
        self.assertEqual(a['fixed_features'],b['fixed_features'])
        self.assertEqual(a['conditioning'],b['conditioning'])
        self.assertEqual(a['structural_null'],b['structural_null'])
        self.assertEqual(b['casting']['solver_dtype'],'torch.float64')
        for arm in (a,b):
            self.assertEqual(arm['state_before'],arm['state_after'])
            for key in ('rng_cpu_unchanged','rng_cuda_unchanged','module_modes_unchanged','existing_gradients_unchanged'):
                self.assertTrue(arm[key])
