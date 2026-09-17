import unittest
from types import SimpleNamespace
import torch
from torch.utils.data import TensorDataset
from pprtp.paired import procrustes,transform,select_anchors,analyze_paired,break_pairs
from pprtp.run import tensor_hash,metrics


class PairedTest(unittest.TestCase):
    def test_pair_breaking_preserves_multiset(self):
        a=torch.arange(4000,dtype=torch.float32).reshape(1000,4)
        broken,r=break_pairs(a,1)
        perm=torch.tensor(r['permutation'])
        self.assertEqual(sorted(r['permutation']),list(range(1000)))
        self.assertTrue(torch.equal(broken[torch.argsort(perm)],a))
        self.assertFalse(torch.equal(broken,a))
        self.assertLessEqual(r['fixed_points'],10)
        self.assertEqual(r,break_pairs(a,1)[1])

    def test_known_rotation_translation(self):
        torch.manual_seed(16)
        x=torch.randn(100,8,dtype=torch.double)
        q=torch.linalg.qr(torch.randn(8,8,dtype=torch.double))[0]
        y=x@q+torch.randn(8,dtype=torch.double)
        t,d=procrustes(x,y)
        torch.testing.assert_close(transform(x,t),y,rtol=1e-10,atol=1e-10)
        self.assertLess(d['orthogonality_error_double'],1e-10)

    def test_label_blind_selection_and_fit(self):
        self.assertEqual(select_anchors(2000,range(200)),select_anchors(2000,range(200)))
        self.assertFalse(set(select_anchors(2000,range(200)))&set(range(200)))
        torch.manual_seed(8)
        clients=[]
        for i in range(2):
            model=torch.nn.Module(); model.base=torch.nn.Linear(10,10);model.head=torch.nn.Linear(10,10)
            clients.append(SimpleNamespace(model=model,device='cpu',class_set=[0,1],protos={0:torch.ones(10)}))
        images=torch.randn(40,10)
        a=TensorDataset(images,torch.zeros(40,dtype=torch.long))
        b=TensorDataset(images,torch.arange(40)*37)
        support=TensorDataset(torch.eye(10),torch.arange(10))
        args=(clients,clients[0].model.head)
        first=analyze_paired(*args,a,[support]*2,support,tensor_hash,metrics)
        second=analyze_paired(*args,b,[support]*2,support,tensor_hash,metrics)
        self.assertEqual(first,second)
        self.assertEqual(first['state_before'],first['state_after'])
