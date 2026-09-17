import inspect
import unittest
import torch
from pprtp.precondition import precondition,apply_precondition


class PreconditionTest(unittest.TestCase):
    def test_invertibility_reconstruction_logits_and_label_isolation(self):
        torch.manual_seed(47)
        x=torch.randn(400,32,dtype=torch.double)*torch.logspace(-3,1,32,dtype=torch.double)
        test=torch.randn(50,32,dtype=torch.double)
        t,inverse,info=precondition(x)
        self.assertEqual(list(inspect.signature(precondition).parameters),['x'])
        self.assertFalse(info['labels_used']);self.assertFalse(info['test_used'])
        self.assertTrue(torch.isfinite(t).all())
        torch.testing.assert_close(t@inverse,torch.eye(32,dtype=torch.double),atol=1e-10,rtol=1e-10)
        for a in (x,test):
            z,error=apply_precondition(a,t,inverse)
            self.assertLess(error,1e-10)
            w=torch.randn(10,32,dtype=torch.double);b=torch.randn(10,dtype=torch.double)
            torch.testing.assert_close(a@w.T+b,z@(w@inverse.T).T+b,rtol=1e-9,atol=1e-9)
            torch.testing.assert_close(z@w.T+b,a@(w@t.T).T+b,rtol=1e-9,atol=1e-9)
        self.assertAlmostEqual(info['condition_after'],1.,places=9)
