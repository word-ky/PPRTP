import contextlib,io,json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from torch.utils.data import TensorDataset,DataLoader
from pprtp.full_data import cifar100_ownership,prepare_cifar100,reserve_anchors,full_readouts
from pprtp.direct_prototypes import analyze_direct
from pprtp.run import metrics,tensor_hash,evaluate,main

class Cifar100Test(unittest.TestCase):
    def test_ownership_label_blind_reservation_and_full_coverage(self):
        sets,order=cifar100_ownership();self.assertEqual(order,np.random.default_rng(120100).permutation(100).tolist())
        self.assertEqual([len(s) for s in sets],[20]*10)
        for j,c in enumerate(order):self.assertEqual([i for i,s in enumerate(sets) if c in s],sorted([j%10,(j+1)%10]))
        labels=np.arange(50000)%100;reserved=[]
        class Train:
            data=np.zeros((50000,1,1,3),dtype=np.uint8)
            @property
            def targets(self):
                assert reserved # labels cannot be accessed before reservation
                return labels
        test=SimpleNamespace(data=np.zeros((10000,1,1,3),dtype=np.uint8),targets=np.arange(10000)%100)
        def reserve(n):
            a=reserve_anchors(n);reserved.extend(a);return a
        def dataset(root,train=True,download=True):return Train() if train else test
        with patch('pprtp.full_data.CIFAR100',side_effect=dataset),patch('pprtp.full_data.reserve_anchors',side_effect=reserve):
            local,evaluation,s,anchors=prepare_cifar100('unused')
        self.assertEqual(s['class_sets'],sets);self.assertEqual(s['train_count'],49744)
        flat=s['anchor_indices']+sum(s['train_indices'],[]);self.assertEqual(sorted(flat),list(range(50000)))
        self.assertEqual(len(evaluation),10000);self.assertTrue(torch.equal(anchors.tensors[1],torch.zeros(256,dtype=torch.long)))
        for c,owners in s['owners'].items():self.assertLessEqual(abs(s['class_counts'][owners[0]][c]-s['class_counts'][owners[1]][c]),1)
        for ds,cs in zip(local,sets):self.assertEqual(sorted(ds.tensors[1].unique().tolist()),cs)

    def test_metrics_histograms_and_all100_rows(self):
        y=torch.arange(100);pred=y.clone();pred[20:]=0
        m=metrics(pred,y,list(range(20)),num_classes=100)
        self.assertEqual(m['seen'],1);self.assertEqual(m['missing'],0);self.assertAlmostEqual(m['all'],.2)
        self.assertEqual(len(m['class_count']),100)
        model=torch.nn.Module();model.base=torch.nn.Identity();model.head=torch.nn.Identity()
        c=SimpleNamespace(device='cpu',class_set=list(range(20)),model=model)
        ds=TensorDataset(torch.eye(100),y);protos={i:torch.eye(100)[i] for i in reversed(range(100))}
        r=evaluate(c,DataLoader(ds,batch_size=32),protos,include_histograms=True,num_classes=100)
        for k in r:self.assertEqual(r[k]['prediction_histogram'],[1]*100);self.assertEqual(r[k]['all'],1)
        self.assertEqual(metrics(torch.arange(10),torch.arange(10),[0,1]),metrics(torch.arange(10),torch.arange(10),[0,1],num_classes=10))

    def test_three_readouts_100class_isolation_and_label_independence(self):
        torch.set_num_threads(1);torch.manual_seed(120);sets,_=cifar100_ownership();local=[];clients=[];vectors=torch.randn(100,16)
        for cs in sets:
            model=torch.nn.Module();model.base=torch.nn.Identity();model.head=torch.nn.Linear(16,100);model.head.eval()
            for p in model.parameters():p.grad=torch.ones_like(p)
            y=torch.tensor(cs).repeat_interleave(3);local.append(TensorDataset(vectors[y]+.01*torch.randn(len(y),16),y))
            clients.append(SimpleNamespace(model=model,device='cpu',class_set=cs,protos={cs[0]:torch.ones(16)}))
        x=torch.randn(256,16);anchors=TensorDataset(x,torch.zeros(256,dtype=torch.long));test=TensorDataset(vectors,torch.arange(100))
        args=(clients,clients[0].model.head,anchors,local,test,tensor_hash,metrics);capture={}
        f=full_readouts(*args,construction_output=capture,num_classes=100);p=f['pprtp_h07'];n=f['native_global_prototype_cosine_control']
        b=analyze_direct(*args,None,broken=True,num_classes=100)
        self.assertEqual(b['anchor_feature_hashes'],capture['anchor_feature_hashes']);self.assertEqual([r['fixed_points'] for r in b['permutation_receipts']],[1,0,1,2,1,0,2,3,1])
        sig=lambda a:[(q['client'],q['label'],q['count'],q['raw_hash']) for q in a['local_prototypes']]
        self.assertEqual(sig(p),sig(n));self.assertEqual(sig(p),sig(b))
        for a in (p,n,b):
            self.assertEqual(a['global_labels'],list(range(100)));self.assertEqual(len(a['prediction_histograms']['total']['overall']),100)
            self.assertEqual(a['state_before'],p['state_before']);self.assertEqual(a['state_before'],a['state_after'])
            for key in ('rng_cpu_unchanged','rng_cuda_unchanged','existing_gradients_unchanged','module_modes_unchanged'):self.assertTrue(a[key])
        changed=analyze_direct(clients,clients[0].model.head,TensorDataset(x,torch.arange(256)%100),local,TensorDataset(vectors,torch.arange(100).roll(1)),tensor_hash,metrics,None,broken=True,num_classes=100)
        for key in ('alignment','global_hash','anchor_feature_hashes'):self.assertEqual(changed[key],b[key])
        self.assertEqual(changed['prediction_histograms']['total']['overall'],b['prediction_histograms']['total']['overall'])

    def test_three_arm_100class_entrypoint(self):
        torch.set_num_threads(1);torch.manual_seed(121);sets,_=cifar100_ownership()
        local=[TensorDataset(torch.randn(20,3,32,32),torch.tensor(cs)) for cs in sets]
        test=TensorDataset(torch.randn(100,3,32,32),torch.arange(100));anchors=TensorDataset(torch.randn(256,3,32,32),torch.zeros(256,dtype=torch.long))
        with tempfile.TemporaryDirectory() as output:
            argv=['pprtp','--data','unused','--output',output,'--device','cpu','--modes','local','fedproto','fedgh','--seeds','0','--rounds','1','--full-data','--dataset','CIFAR100','--num-classes','100','--k','20']
            with patch('sys.argv',argv),patch('pprtp.full_data.prepare_cifar100',return_value=(local,test,dict(class_sets=sets),anchors)),contextlib.redirect_stdout(io.StringIO()):main()
            f=json.loads((Path(output)/'fedgh_seed0/final.json').read_text());self.assertEqual(f['local_optimizer_steps'],[1]*10)
            self.assertEqual(len(f['prediction_histograms']['global_head_post_server']),100)
            self.assertTrue(f['full_pair_probe']['same_raw_means_counts_exact']);self.assertIsNone(f['full_pair_probe']['h11_entire_reference_exact'])
            self.assertEqual(f['communication_bytes']['upload_vectors'],200*512*4)
            self.assertTrue(json.loads((Path(output)/'round_one_pairing_seed0.json').read_text())['passed'])
