import unittest
import torch
from pprtp.centered import centered_scores
from pprtp.direct_prototypes import cosine_scores


class CenteredTest(unittest.TestCase):
    def test_invariance_and_exact_assembly(self):
        torch.manual_seed(93);z=torch.randn(13,7);p=torch.randn(2,7);labels=torch.tensor([2,8]);g=torch.randn(10,7)
        q=torch.linalg.qr(torch.randn(7,7)).Q;mu=torch.randn(7);ref=torch.randn(7)
        scores,receipt=centered_scores(z,p,labels,g,(mu,q,ref))
        self.assertLess(receipt['owner_rotation_max_abs_error'],2e-5)
        self.assertTrue(torch.equal(scores[:,labels],cosine_scores(z-mu,p-mu)))
        missing=[i for i in range(10) if i not in labels]
        self.assertTrue(torch.equal(scores[:,missing],cosine_scores((z-mu)@q,g-ref)[:,missing]))
        native_shift=torch.randn(7)*5;ref_shift=torch.randn(7)*8
        moved,_=centered_scores(z+native_shift,p+native_shift,labels,g+ref_shift,(mu+native_shift,q,ref+ref_shift))
        torch.testing.assert_close(scores,moved,atol=2e-5,rtol=2e-5)
        control,_=centered_scores(z,p,labels,g,(mu,q,ref),True)
        self.assertTrue(torch.equal(control,cosine_scores((z-mu)@q,g-ref)))
