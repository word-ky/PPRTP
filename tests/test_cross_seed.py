import json,unittest
from pathlib import Path
import numpy as np
from pprtp.data import partition
from pprtp.cross_seed import construct_indices,index_hash


class CrossSeedTest(unittest.TestCase):
    def test_seed0_historical_regeneration(self):
        labels=np.frombuffer(Path('research_log/H04B/cifar10-train-labels.bin').read_bytes(),dtype=np.uint8)
        sets,indices=partition(labels,0)
        receipt=construct_indices(labels,dict(class_sets=sets,train_indices=indices))
        oracle=json.loads(Path('research_log/H02C/full/artifacts/experiment/fedgh_seed0/oracle_calibration.json').read_text())
        owner=json.loads(Path('research_log/H02E/full/artifacts/experiment/fedgh_seed0/heldout_owner_support.json').read_text())
        anchor=json.loads(Path('research_log/H03A/gate/artifacts/experiment/fedgh_seed0/paired_anchors.json').read_text())
        self.assertEqual(receipt['oracle_indices'],oracle['indices'])
        self.assertEqual(receipt['support_indices'],owner['indices'])
        self.assertEqual(receipt['anchor_indices'],anchor['indices'])
        self.assertEqual(receipt['oracle_sha256'],oracle['index_sha256'])
        self.assertEqual(receipt['support_sha256'],owner['indices_sha256'])
        self.assertEqual(receipt['anchor_sha256'],anchor['indices_sha256'])

    def test_seed_specific_disjoint_construction(self):
        labels=np.frombuffer(Path('research_log/H04B/cifar10-train-labels.bin').read_bytes(),dtype=np.uint8)
        receipts=[]
        for seed in (0,1,2):
            sets,indices=partition(labels,seed);split=dict(class_sets=sets,train_indices=indices)
            r=construct_indices(labels,split)
            self.assertEqual(r,construct_indices(labels,split))
            groups=[set(sum(indices,[])),set(r['oracle_indices']),set(sum(r['support_indices'],[])),set(r['anchor_indices'])]
            self.assertEqual([len(g) for g in groups],[2000,1000,2000,1000])
            self.assertTrue(all(not a&b for i,a in enumerate(groups) for b in groups[i+1:]))
            receipts.append(r)
        self.assertEqual(len({r['anchor_sha256'] for r in receipts}),3)
        self.assertEqual(len({r['support_sha256'] for r in receipts}),3)
