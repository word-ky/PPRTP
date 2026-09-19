import contextlib,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import TensorDataset
from pprtp.run import main


class TinyRunnerTest(unittest.TestCase):
    def test_64pixel_four_arm_pairing_and_200class_readouts(self):
        torch.set_num_threads(1);torch.manual_seed(170)
        order=np.random.default_rng(120200).permutation(200).tolist();sets=[sorted(order[i::10]) for i in range(10)]
        local=[TensorDataset(torch.randn(20,3,64,64),torch.tensor(cs)) for cs in sets]
        val=TensorDataset(torch.randn(200,3,64,64),torch.arange(200));anchors=TensorDataset(torch.randn(256,3,64,64),torch.zeros(256,dtype=torch.long))
        with tempfile.TemporaryDirectory() as output:
            argv=['pprtp','--data','unused','--output',output,'--device','cpu','--modes','local','fedproto','fedgh','fedavg','--seeds','0','--rounds','1','--full-data','--dataset','TinyImageNet','--num-classes','200','--k','20','--owners-per-class','1','--ownership-seed','120200']
            with patch('sys.argv',argv),patch('pprtp.tiny_data.prepare_tiny',return_value=(local,val,dict(class_sets=sets),anchors)),contextlib.redirect_stdout(io.StringIO()):main()
            base=Path(output)
            self.assertTrue(json.loads((base/'round_one_pairing_seed0.json').read_text())['passed'])
            for mode in ('local','fedproto','fedgh','fedavg'):
                m=json.loads((base/f'{mode}_seed0/metadata.json').read_text());r=json.loads((base/f'{mode}_seed0/final.json').read_text())
                self.assertEqual((m['input_size'],m['feature_dim'],m['head_classes'],m['cnn_dim']),(64,512,200,10816))
                self.assertTrue(m['model_shape_checked']);self.assertFalse(m['pretrained']);self.assertEqual(r['local_optimizer_steps'],[1]*10)
                for client in r['per_client']:
                    for metric in client.values():self.assertEqual(metric['class_count'],[1]*200)
            f=json.loads((base/'fedgh_seed0/final.json').read_text());paired=f['full_data_readout']['pprtp_h07'];native=f['full_data_readout']['native_global_prototype_cosine_control'];broken=f['full_pair_probe']['pair_broken_h07']
            self.assertTrue(f['full_pair_probe']['same_raw_means_counts_exact']);self.assertTrue(f['full_pair_probe']['anchor_feature_hashes_exact'])
            for arm in (paired,native,broken):
                self.assertEqual(arm['global_labels'],list(range(200)));self.assertEqual(arm['state_before'],arm['state_after'])
                self.assertTrue(arm['cosine_logits_finite']);self.assertFalse(arm['test_used_for_transform']);self.assertFalse(arm['anchor_labels_used'])
            avg=json.loads((base/'fedavg_seed0/final.json').read_text());self.assertFalse(avg['fedavg_server']['anchors_used'])
            hh=[v['global_model_post_server']['prediction_histogram'] for v in avg['per_client']];self.assertTrue(all(v==hh[0] for v in hh))
