import contextlib,io,json,tempfile,unittest,gc
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch.utils.data import TensorDataset
from pprtp.mixed_backbone import build_mixed
from pprtp.run import main,tensor_hash
from pprtp.oracle import features

class MixedBackboneTest(unittest.TestCase):
    def test_deterministic_construction_shapes_and_batchnorm_isolation(self):
        torch.set_num_threads(1)
        models,a=build_mixed(0,100,tensor_hash)
        self.assertEqual([r['architecture'] for r in a],['FedAvgCNN','ResNet18']*5)
        self.assertNotEqual(a[0]['initial_model_hash'],a[2]['initial_model_hash'])
        self.assertTrue(any('running_mean' in k for k in a[1]['buffer_names']))
        model=models[1];model.train();model.head.eval()
        before=tensor_hash(model.state_dict().values());modes=[m.training for m in model.modules()]
        ds=TensorDataset(torch.randn(4,3,32,32),torch.arange(4))
        z,_=features(SimpleNamespace(model=model,device='cpu'),ds)
        self.assertEqual(z.shape,(4,512));self.assertEqual(tensor_hash(model.state_dict().values()),before)
        self.assertEqual([m.training for m in model.modules()],modes)
        del model,models;gc.collect()
        models,b=build_mixed(0,100,tensor_hash);self.assertEqual(a,b)

    def test_mixed_three_arm_entrypoint_and_readout_isolation(self):
        torch.set_num_threads(1);torch.manual_seed(130)
        split=json.loads(Path('research_log/H12A/full/artifacts/experiment/local_seed0/split.json').read_text())
        local=[TensorDataset(torch.randn(20,3,32,32),torch.tensor(cs)) for cs in split['class_sets']]
        evaluation=TensorDataset(torch.randn(100,3,32,32),torch.arange(100));anchors=TensorDataset(torch.randn(256,3,32,32),torch.zeros(256,dtype=torch.long))
        with tempfile.TemporaryDirectory() as output:
            argv=['pprtp','--data','unused','--output',output,'--device','cpu','--modes','local','fedproto','fedgh','--seeds','0','--rounds','1','--full-data','--dataset','CIFAR100','--num-classes','100','--k','20','--mixed-backbone']
            with patch('sys.argv',argv),patch('pprtp.full_data.prepare_cifar100',return_value=(local,evaluation,split,anchors)),contextlib.redirect_stdout(io.StringIO()):main()
            read=lambda mode,name:json.loads((Path(output)/f'{mode}_seed0'/name).read_text())
            mm=[read(m,'metadata.json') for m in ('local','fedproto','fedgh')]
            self.assertEqual(mm[0]['client_initial_states'],mm[1]['client_initial_states']);self.assertEqual(mm[0]['client_initial_states'],mm[2]['client_initial_states'])
            rr=[read(m,'final.json') for m in ('local','fedproto','fedgh')]
            self.assertEqual(rr[0]['batch_hashes'],rr[1]['batch_hashes']);self.assertEqual(rr[0]['batch_hashes'],rr[2]['batch_hashes'])
            self.assertEqual(rr[0]['client_model_hashes'],rr[2]['client_model_hashes'])
            f=rr[2]['full_data_readout'];b=rr[2]['full_pair_probe']['pair_broken_h07'];p=f['pprtp_h07'];n=f['native_global_prototype_cosine_control']
            for r in (p,n,b):
                self.assertEqual(r['state_before'],r['state_after']);self.assertEqual(r['state_before'],p['state_before'])
                for k in ('rng_cpu_unchanged','rng_cuda_unchanged','module_modes_unchanged','existing_gradients_unchanged'):self.assertTrue(r[k])
            self.assertTrue(rr[2]['full_pair_probe']['same_raw_means_counts_exact']);self.assertTrue(rr[2]['full_pair_probe']['anchor_feature_hashes_exact'])
            self.assertTrue(json.loads((Path(output)/'round_one_pairing_seed0.json').read_text())['passed'])

    def test_seed_replication_initialization_and_actual_batch_orders(self):
        torch.set_num_threads(1);torch.manual_seed(130)
        split=json.loads(Path('research_log/H12A/full/artifacts/experiment/local_seed0/split.json').read_text())
        local=[TensorDataset(torch.randn(40,3,32,32),torch.tensor(cs*2)) for cs in split['class_sets']]
        evaluation=TensorDataset(torch.randn(100,3,32,32),torch.arange(100))
        anchors=TensorDataset(torch.randn(256,3,32,32),torch.zeros(256,dtype=torch.long))
        with tempfile.TemporaryDirectory() as output:
            argv=['pprtp','--data','unused','--output',output,'--device','cpu','--modes','local','--seeds','0','1','2','--rounds','1','--full-data','--dataset','CIFAR100','--num-classes','100','--k','20','--mixed-backbone']
            with patch('sys.argv',argv),patch('pprtp.full_data.prepare_cifar100',return_value=(local,evaluation,split,anchors)),contextlib.redirect_stdout(io.StringIO()):main()
            roots=[Path(output)/f'local_seed{seed}' for seed in (0,1,2)]
            initial=[json.loads((r/'metadata.json').read_text())['client_initial_states'] for r in roots]
            batches=[json.loads((r/'final.json').read_text())['batch_hashes'] for r in roots]
            for r in roots:self.assertEqual(json.loads((r/'split.json').read_text()),split)
            for i in range(10):
                self.assertEqual(len({rr[i]['initial_model_hash'] for rr in initial}),3)
                self.assertEqual(len({tuple(bb[i]) for bb in batches}),3)
                for rr in initial:
                    self.assertEqual(rr[i]['architecture'],'FedAvgCNN' if i%2==0 else 'ResNet18')
                    self.assertEqual(rr[i]['head_shapes'],{'weight':[100,512],'bias':[100]})
            models,repeated=build_mixed(1,100,tensor_hash)
            self.assertEqual(initial[1],repeated)
