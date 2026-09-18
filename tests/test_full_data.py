import inspect,json,unittest
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import TensorDataset
from types import SimpleNamespace
from pprtp.full_data import reserve_anchors,allocate,full_readouts
from pprtp.run import tensor_hash,metrics

class FullDataTest(unittest.TestCase):
    def test_partition_full_coverage_and_frozen_ownership(self):
        sets=json.loads(Path('research_log/H02A/full/artifacts/experiment/fedgh_seed0/split.json').read_text())['class_sets']
        labels=np.repeat(np.arange(10),5000);anchors=reserve_anchors(len(labels))
        self.assertEqual(list(inspect.signature(reserve_anchors).parameters),['size'])
        self.assertEqual(len(set(anchors)),256);self.assertEqual(anchors,reserve_anchors(len(labels)))
        a=allocate(labels,sets,anchors);b=allocate(labels[::-1],sets,anchors)
        self.assertEqual(a['anchor_indices_sha256'],b['anchor_indices_sha256'])
        self.assertEqual(a['train_count'],49744);self.assertEqual(a['coverage_count'],50000)
        self.assertTrue(all(len(o)==2 for o in a['owners'].values()));self.assertEqual(a,allocate(labels,sets,anchors))
        for i,ii in enumerate(a['train_indices']):self.assertEqual(sorted(np.unique(labels[ii]).tolist()),sets[i])

    def test_readout_isolation_counts_and_anchor_label_blindness(self):
        torch.manual_seed(111);clients=[];local=[]
        for i in range(10):
            model=torch.nn.Module();model.base=torch.nn.Identity();model.head=torch.nn.Linear(10,10)
            model.head.eval()
            for p in model.parameters():p.grad=torch.ones_like(p)
            cs=sorted([i,(i+1)%10]);y=torch.tensor([cs[0]]*(5+i)+[cs[1]]*(7+i))
            local.append(TensorDataset(torch.eye(10)[y]+.02*torch.randn(len(y),10),y))
            clients.append(SimpleNamespace(model=model,device='cpu',class_set=cs,protos={cs[0]:torch.ones(10)}))
        x=torch.randn(256,10);anchors=TensorDataset(x,torch.zeros(256,dtype=torch.long));test=TensorDataset(torch.eye(10),torch.arange(10))
        args=(clients,clients[0].model.head,anchors,local,test,tensor_hash,metrics)
        a=full_readouts(*args)
        b=full_readouts(clients,clients[0].model.head,TensorDataset(x,torch.randint(10,(256,))),local,test,tensor_hash,metrics)
        self.assertEqual(a,b);self.assertTrue(a['same_raw_means_counts_exact'])
        self.assertEqual(a['forward_examples']['pprtp_total'],2560+sum(map(len,local)))
        self.assertEqual(a['communication']['naive_affine_downlink_per_client'],[480]*10)

    def test_step_receipt_and_optional_histograms_do_not_change_metrics(self):
        from test_baseline import fixture
        from pprtp.client import H01Client
        from pprtp.run import evaluate
        c=fixture(H01Client);c.train()
        self.assertEqual(c.optimizer_steps,2)
        from torch.utils.data import DataLoader
        loader=DataLoader(TensorDataset(torch.randn(10,3,32,32),torch.arange(10)),batch_size=4)
        a=evaluate(c,loader,c.protos)
        b=evaluate(c,loader,c.protos,include_histograms=True)
        for k in a:
            self.assertEqual(sum(b[k].pop('prediction_histogram')),10)
            b[k].pop('predicted_class_count')
        self.assertEqual(a,b)

    def test_three_arm_entrypoint_and_final_readouts(self):
        import tempfile,contextlib,io
        from unittest.mock import patch
        from pprtp.run import main
        torch.set_num_threads(1);torch.manual_seed(112)
        sets=json.loads(Path('research_log/H02A/full/artifacts/experiment/fedgh_seed0/split.json').read_text())['class_sets']
        local=[TensorDataset(torch.randn(4,3,32,32),torch.tensor(cs*2)) for cs in sets]
        test=TensorDataset(torch.randn(10,3,32,32),torch.arange(10))
        anchors=TensorDataset(torch.randn(16,3,32,32),torch.zeros(16,dtype=torch.long))
        split=dict(class_sets=sets)
        with tempfile.TemporaryDirectory() as output:
            argv=['pprtp','--data','unused','--output',output,'--device','cpu','--modes','local','fedproto','fedgh','--seeds','0','--rounds','1','--full-data']
            with patch('sys.argv',argv),patch('pprtp.full_data.prepare_full',return_value=(local,test,split,anchors)),contextlib.redirect_stdout(io.StringIO()):main()
            r=json.loads((Path(output)/'fedgh_seed0/final.json').read_text())
            self.assertIsNone(r['historical_round_one_paired']);self.assertEqual(r['local_optimizer_steps'],[1]*10)
            self.assertTrue(r['full_data_readout']['same_final_state_exact'])
            self.assertEqual(sum(r['prediction_histograms']['global_head_post_server']),100)
            self.assertTrue(json.loads((Path(output)/'round_one_pairing_seed0.json').read_text())['passed'])

    def test_seed_generalization_and_exact_h11a_split(self):
        from unittest.mock import patch
        from pprtp.full_data import prepare_full
        old=json.loads(Path('research_log/H11A/full/artifacts/experiment/fedgh_seed0/split.json').read_text())
        labels=np.zeros(50000,dtype=np.int64)
        for ii,counts,cs in zip(old['train_indices'],old['class_counts'],old['class_sets']):
            offset=0
            for c in cs:
                n=counts[str(c)];labels[ii[offset:offset+n]]=c;offset+=n
        train=SimpleNamespace(data=np.zeros((50000,1,1,3),dtype=np.uint8),targets=labels)
        test=SimpleNamespace(data=np.zeros((10000,1,1,3),dtype=np.uint8),targets=np.arange(10000)%10)
        def dataset(root,train=True,download=True):return fixtures[train]
        fixtures={True:train,False:test}
        for seed in (0,1,2):
            with patch('pprtp.full_data.CIFAR10',side_effect=dataset):
                local,evaluation,split,anchors=prepare_full('unused',seed)
            history=json.loads(Path(f'research_log/H04B/full/artifacts/experiment/fedgh_seed{seed}/split.json').read_text()) if seed else old
            self.assertEqual(split['class_sets'],history['class_sets'])
            self.assertEqual(split['anchor_indices'],old['anchor_indices'])
            self.assertEqual(sum(map(len,local)),49744);self.assertEqual(len(evaluation),10000)
            self.assertTrue(torch.equal(anchors.tensors[1],torch.zeros(256,dtype=torch.long)))
            if seed==0:self.assertEqual(json.loads(json.dumps(split)),old)
