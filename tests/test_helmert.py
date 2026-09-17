import unittest
import torch
from pprtp.helmert import basis,null_diagnostics


class HelmertTest(unittest.TestCase):
    def test_basis_reconstruction_and_head_equivalence(self):
        torch.manual_seed(45);q=basis(256,dtype=torch.double)
        torch.testing.assert_close(q.T@q,torch.eye(255,dtype=torch.double),atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(q.T@torch.ones(256,dtype=torch.double),torch.zeros(255,dtype=torch.double),atol=1e-12,rtol=0)
        r=torch.randn(20,256,dtype=torch.double);r-=r.mean(1,keepdim=True)
        torch.testing.assert_close(r,(r@q)@q.T,atol=1e-12,rtol=1e-12)
        w=torch.randn(10,256,dtype=torch.double);b=torch.randn(10,dtype=torch.double)
        torch.testing.assert_close(r@w.T+b,(r@q)@(w@q).T+b,atol=1e-11,rtol=1e-11)
        u=torch.randn(10,255,dtype=torch.double)
        torch.testing.assert_close((r@q)@u.T,r@(u@q.T).T,atol=1e-11,rtol=1e-11)

    def test_known_null_energy(self):
        x=torch.tensor([[2.,0.],[0.,2.]])
        d=null_diagnostics(x)
        self.assertEqual(d['row_sum_abs_max'],2.)
        self.assertEqual(d['row_sum_rms'],2.)
        self.assertEqual(d['ones_energy_fraction'],.5)
        self.assertEqual(null_diagnostics(torch.tensor([[1.,-1.]]))['ones_energy_fraction'],0.)
