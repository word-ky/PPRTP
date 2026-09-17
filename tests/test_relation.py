import unittest
import torch
from pprtp.relation import relation,permutation,permute_relations


class RelationTest(unittest.TestCase):
    def test_orthogonal_translation_invariance(self):
        torch.manual_seed(41)
        anchors=torch.randn(256,16,dtype=torch.double);z=torch.randn(30,16,dtype=torch.double)
        q=torch.linalg.qr(torch.randn(16,16,dtype=torch.double))[0];shift=torch.randn(16,dtype=torch.double)
        torch.testing.assert_close(relation(z,anchors),relation(z@q+shift,anchors@q+shift),rtol=1e-10,atol=1e-10)

    def test_column_permutation_preserves_each_sample(self):
        r=torch.arange(1024,dtype=torch.float32).reshape(4,256)
        for i in range(10):
            receipt=permutation(i);p=receipt['permutation'];out=permute_relations(r,p)
            self.assertEqual(sorted(p),list(range(256)))
            self.assertTrue(torch.equal(out[:,torch.argsort(torch.tensor(p))],r))
            self.assertEqual(receipt,permutation(i))
            if i: self.assertFalse(torch.equal(out,r))
            else: self.assertTrue(torch.equal(out,r))
