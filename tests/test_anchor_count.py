import unittest
from types import SimpleNamespace
import torch
from torch.utils.data import TensorDataset
from pprtp.anchor_count import prefix,COUNTS,historical_fields
from pprtp.paired import analyze_paired,procrustes
from pprtp.run import tensor_hash,metrics


class AnchorCountTest(unittest.TestCase):
    def test_nested_prefixes(self):
        indices=list(reversed(range(1000)))
        data=TensorDataset(torch.arange(1000)[:,None],torch.zeros(1000))
        for n in COUNTS:
            ds,r=prefix(data,indices,n)
            self.assertEqual(r['indices'],indices[:n])
            self.assertTrue(torch.equal(ds.tensors[0],data.tensors[0][:n]))
            self.assertEqual(r,prefix(data,indices,n)[1])

    def test_rank_diagnostics_preserve_full_anchor_result(self):
        torch.manual_seed(34)
        model=torch.nn.Module();model.base=torch.nn.Identity();model.head=torch.nn.Linear(10,10)
        client=SimpleNamespace(model=model,device='cpu',class_set=[0,1],protos={0:torch.ones(10)})
        anchors=TensorDataset(torch.randn(1000,10),torch.zeros(1000,dtype=torch.long))
        support=TensorDataset(torch.eye(10),torch.arange(10))
        args=([client],model.head,anchors,[support],support,tensor_hash,metrics)
        old=analyze_paired(*args,max_iter=2000,audit=True)
        new=analyze_paired(*args,max_iter=2000,audit=True,rank_diagnostics=True)
        self.assertEqual(old,historical_fields(new))
        x=torch.randn(4,10)
        _,d=procrustes(x,x,rank_diagnostics=True)
        self.assertEqual(d['rank']['centered_rank_ceiling'],3)
        self.assertLessEqual(d['rank']['effective_rank'],3)
        self.assertTrue(all(torch.isfinite(torch.tensor(d['rank'][k])) for k in
            ('tolerance','largest_singular_value','smallest_nonzero_singular_value')))
