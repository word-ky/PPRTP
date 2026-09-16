import unittest
import numpy as np
import torch
from pprtp.data import partition
from pprtp.run import metrics


class DataMetricTests(unittest.TestCase):
    def test_partition_disjoint_complete_and_reproducible(self):
        labels=np.repeat(np.arange(10),50)
        a=partition(labels,0,10,2,20)
        self.assertEqual(a,partition(labels,0,10,2,20))
        sets,idx=a
        flat=sum(idx,[])
        self.assertEqual(len(flat),len(set(flat)))
        self.assertEqual(set(sum(sets,[])),set(range(10)))
        for classes,indices in zip(sets,idx):
            self.assertEqual(len(classes),2)
            self.assertEqual(set(labels[indices]),set(classes))

    def test_known_metrics(self):
        labels=torch.arange(10)
        pred=labels.clone(); pred[2:]=0
        result=metrics(pred,labels,[0,1])
        self.assertEqual(result['seen'],1.)
        self.assertEqual(result['missing'],0.)
        self.assertAlmostEqual(result['all'],.2)
