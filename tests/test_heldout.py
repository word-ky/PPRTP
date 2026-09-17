import unittest
import numpy as np
from pprtp.heldout import assign_indices


class HeldoutTest(unittest.TestCase):
    def test_assignment_counts_support_disjoint_and_deterministic(self):
        labels=np.repeat(np.arange(10),600)
        split=dict(class_sets=[sorted([i,(i+1)%10]) for i in range(10)],
                   train_indices=[list(range(i*600,i*600+200)) for i in range(10)])
        oracle=[i*600+j for i in range(10) for j in range(200,300)]
        a=assign_indices(labels,split,oracle)
        self.assertEqual(a,assign_indices(labels,split,oracle))
        flat=sum(a,[])
        self.assertEqual(len(set(flat)),2000)
        self.assertFalse(set(flat)&(set(sum(split['train_indices'],[]))|set(oracle)))
        self.assertEqual(np.bincount(labels[flat]).tolist(),[200]*10)
        for ii,ss in zip(a,split['class_sets']):
            self.assertEqual(len(ii),200)
            self.assertEqual(sorted(set(labels[ii])),ss)
