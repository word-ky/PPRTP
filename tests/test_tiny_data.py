import tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor,normalize
from pprtp.tiny_data import index_tiny,parse_validation,load_images,prepare_tiny
from pprtp.full_data import reserve_anchors


class TinyDataTest(unittest.TestCase):
    def fixture(self,root):
        (root/'wnids.txt').write_text('n002\nn001\n')
        for wnid in ('n002','n001'):
            folder=root/'train'/wnid/'images';folder.mkdir(parents=True)
            for i in range(2):Image.new('RGB',(64,64),(i*100,50,200)).save(folder/f'{wnid}_{i}.JPEG')
        folder=root/'val/images';folder.mkdir(parents=True)
        Image.new('L',(64,64),80).save(folder/'val_0.JPEG');Image.new('RGB',(64,64),(30,20,10)).save(folder/'val_1.JPEG')
        (root/'val/val_annotations.txt').write_text('val_1.JPEG n001 0 0 63 63\nval_0.JPEG n002 0 0 63 63\n')

    def test_lexical_mapping_annotations_and_label_blind_anchor_order(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.fixture(root);reserved=[]
            def reserve(size):
                self.assertEqual(size,4);a=reserve_anchors(size);reserved.extend(a);return a
            def val(*args):
                self.assertTrue(reserved);return parse_validation(*args)
            with patch('pprtp.tiny_data.reserve_anchors',side_effect=reserve),patch('pprtp.tiny_data.parse_validation',side_effect=val):
                paths,y,vp,vy,aa,m=index_tiny(root,2,2,1)
            self.assertEqual(m['class_to_idx'],{'n001':0,'n002':1});self.assertEqual(y.tolist(),[0,0,1,1]);self.assertEqual(vy.tolist(),[1,0])
            self.assertEqual(aa,reserve_anchors(4));self.assertEqual(len(m['train_file_hashes']),4)
            self.assertFalse(set(m['train_files'])&set(m['val_files']));self.assertFalse(m['official_test_used'])
            with Image.open(vp[0]) as img:expected=normalize(to_tensor(img.convert('RGB')),[.5]*3,[.5]*3)
            self.assertTrue(torch.equal(load_images(vp)[0],expected))
            (root/'val/val_annotations.txt').write_text('val_0.JPEG n001\nval_0.JPEG n002\n')
            with self.assertRaises(AssertionError):parse_validation(root,m['class_to_idx'],1)

    def test_full_one_owner_coverage_without_image_memory(self):
        labels=np.repeat(np.arange(200),500);anchors=reserve_anchors(100000)
        manifest=dict(raw_train_count=100000,raw_val_count=10000,raw_train_class_counts=[500]*200,raw_val_class_counts=[50]*200)
        def pixels(paths):return torch.zeros(len(paths),3,1,1)
        with patch('pprtp.tiny_data.index_tiny',return_value=(list(range(100000)),labels,list(range(10000)),np.repeat(np.arange(200),50),anchors,manifest)),patch('pprtp.tiny_data.load_images',side_effect=pixels):
            local,test,s,public=prepare_tiny('unused')
        self.assertEqual(sorted(anchors+sum(s['train_indices'],[])),list(range(100000)))
        self.assertEqual(s['train_count'],99744);self.assertEqual([len(cs) for cs in s['class_sets']],[20]*10)
        self.assertEqual(s['ownership_order'],np.random.default_rng(120200).permutation(200).tolist())
        for j,c in enumerate(s['ownership_order']):self.assertEqual(s['owners'][c],[j%10])
        for ds,cs in zip(local,s['class_sets']):self.assertEqual(sorted(ds.tensors[1].unique().tolist()),cs)
        self.assertEqual(len(test),10000);self.assertTrue(torch.equal(public.tensors[1],torch.zeros(256,dtype=torch.long)))
