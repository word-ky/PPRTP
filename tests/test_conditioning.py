import unittest
import torch
from pprtp.conditioning import statistics,apply_condition,raw_head,matrix_diagnostics


class ConditioningTest(unittest.TestCase):
    def test_affine_logits_equivalence(self):
        torch.manual_seed(43)
        support=torch.randn(100,16,dtype=torch.double)*torch.arange(1,17,dtype=torch.double)+3
        test=torch.randn(30,16,dtype=torch.double)
        mean,std=statistics(support)
        head=torch.nn.Linear(16,10,dtype=torch.double)
        w,b=raw_head(head.weight,head.bias,mean,std)
        torch.testing.assert_close(head(apply_condition(test,mean,std)),test@w.T+b,rtol=1e-12,atol=1e-12)
        conditioned=apply_condition(support,mean,std)
        torch.testing.assert_close(conditioned.mean(0),torch.zeros_like(mean),atol=1e-12,rtol=0)
        torch.testing.assert_close(conditioned.square().mean(0),torch.ones_like(std),atol=1e-12,rtol=0)
        self.assertTrue(matrix_diagnostics(conditioned)['finite'])
        with self.assertRaises(AssertionError): statistics(torch.ones(5,3))
