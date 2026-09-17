import unittest
import torch
from pprtp.completion import complete


class CompletionTest(unittest.TestCase):
    def test_nullspace_freedom_and_local_rng(self):
        torch.manual_seed(48)
        x=torch.randn(20,3,dtype=torch.double)@torch.randn(3,8,dtype=torch.double);x-=x.mean(0)
        y=torch.randn(20,3,dtype=torch.double)@torch.randn(3,8,dtype=torch.double);y-=y.mean(0)
        u,s,vh=torch.linalg.svd(x.T@y);canonical=u@vh
        torch.testing.assert_close(canonical.T@canonical,torch.eye(8,dtype=torch.double),atol=1e-12,rtol=1e-12)
        rng=torch.get_rng_state().clone()
        for seed in (602101,602201,602301):
            r,d=complete(x,y,u,s,vh,seed,rank=3)
            torch.testing.assert_close(x@r,x@canonical,atol=1e-10,rtol=1e-10)
            torch.testing.assert_close(r.T@r,torch.eye(8,dtype=torch.double),atol=1e-12,rtol=1e-12)
            self.assertGreater((u[:,3:].T@r-u[:,3:].T@canonical).norm().item(),1.)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertTrue(torch.equal(r,complete(x,y,u,s,vh,seed,rank=3)[0]))
