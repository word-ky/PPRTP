import unittest
from types import SimpleNamespace
import pprtp
import torch
from pprtp.client import knowledge_loss
from pprtp.run import owner_compatibility


class H01CTests(unittest.TestCase):
    def test_denominator_and_detachment(self):
        torch.manual_seed(7)
        z=torch.randn(4,5,requires_grad=True)
        y=torch.tensor([0,1,0,1])
        bank=torch.randn(10,5,requires_grad=True)
        changed=bank.detach().clone(); changed[8]=z[0].detach()
        valid=torch.ones(10,dtype=torch.bool)
        for mode in ('gpc_all_match','gpc_seen_match'):
            a=knowledge_loss(z,y,bank,valid,mode,10.,[0,1])
            b=knowledge_loss(z,y,changed,valid,mode,10.,[0,1])
            ga=torch.autograd.grad(a,z,retain_graph=True)[0]
            gb=torch.autograd.grad(b,z,retain_graph=True)[0]
            if mode=='gpc_all_match':
                self.assertGreater(abs(a.item()-b.item()),1e-5)
                self.assertGreater((ga-gb).abs().max().item(),1e-5)
            else:
                torch.testing.assert_close(a,b,rtol=0,atol=0)
                torch.testing.assert_close(ga,gb,rtol=0,atol=0)
                expected=torch.nn.functional.cross_entropy(
                    10*torch.nn.functional.normalize(z,dim=1) @ torch.nn.functional.normalize(bank[:2].detach(),dim=1).T,y)
                torch.testing.assert_close(a,expected)
            self.assertIsNone(torch.autograd.grad(a,bank,allow_unused=True)[0])

    def test_owner_cosine(self):
        a=SimpleNamespace(id=0,protos={0:torch.tensor([1.,0.]),1:torch.tensor([0.,1.])})
        b=SimpleNamespace(id=1,protos={0:torch.tensor([1.,0.]),1:torch.tensor([1.,0.])})
        d=owner_compatibility([a,b])
        self.assertEqual(d['mean'],.5)
        self.assertEqual(d['min'],0.)
        self.assertEqual(d['max'],1.)
