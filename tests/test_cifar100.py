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


    def test_training_seeds_keep_h12a_split_and_change_initialization(self):
        old=json.loads(Path('research_log/H12A/full/artifacts/experiment/local_seed0/split.json').read_text())
        labels=np.zeros(50000,dtype=np.int64)
        for ii,counts,cs in zip(old['train_indices'],old['class_counts'],old['class_sets']):
            offset=0
            for c in cs:
                n=counts[str(c)];labels[ii[offset:offset+n]]=c;offset+=n
        train=SimpleNamespace(data=np.zeros((50000,1,1,3),dtype=np.uint8),targets=labels)
        test=SimpleNamespace(data=np.zeros((10000,1,1,3),dtype=np.uint8),targets=np.arange(10000)%100)
        for seed in (0,1,2):
            torch.manual_seed(seed);np.random.seed(seed)
            with patch('pprtp.full_data.CIFAR100',side_effect=lambda root,train=True,download=True:fixtures[train]):
                fixtures={True:train,False:test};_,_,split,_=prepare_cifar100('unused')
            self.assertEqual(json.loads(json.dumps(split)),old)
        torch.manual_seed(122);sets=old['class_sets'];local=[TensorDataset(torch.randn(40,3,32,32),torch.tensor(cs*2)) for cs in sets]
        evaluation=TensorDataset(torch.randn(100,3,32,32),torch.arange(100));anchors=TensorDataset(torch.randn(256,3,32,32),torch.zeros(256,dtype=torch.long))
        with tempfile.TemporaryDirectory() as output:
            argv=['pprtp','--data','unused','--output',output,'--device','cpu','--modes','local','--seeds','0','1','2','--rounds','1','--full-data','--dataset','CIFAR100','--num-classes','100','--k','20']
            with patch('sys.argv',argv),patch('pprtp.full_data.prepare_cifar100',return_value=(local,evaluation,old,anchors)),contextlib.redirect_stdout(io.StringIO()):main()
            meta=[json.loads((Path(output)/f'local_seed{i}/metadata.json').read_text()) for i in (0,1,2)]
            self.assertEqual(len({m['initial_state_sha256'] for m in meta}),3)
            self.assertEqual(len({m['split_sha256'] for m in meta}),1)
            final=[json.loads((Path(output)/f'local_seed{i}/final.json').read_text()) for i in (0,1,2)]
            self.assertEqual(len({f['client_model_hashes'][0] for f in final}),3)
            orders=[torch.randperm(40,generator=torch.Generator().manual_seed(seed*100000)).tolist() for seed in (0,1,2)]
            self.assertEqual(len({tuple(o) for o in orders}),3)

    def test_explicit_ownership_seed_reproduces_old_split_and_changes_graph(self):
        old=json.loads(Path('research_log/H12A/full/artifacts/experiment/local_seed0/split.json').read_text())
        labels=np.zeros(50000,dtype=np.int64)
        for ii,counts,cs in zip(old['train_indices'],old['class_counts'],old['class_sets']):
            offset=0
            for c in cs:
                n=counts[str(c)];labels[ii[offset:offset+n]]=c;offset+=n
        fixtures={True:SimpleNamespace(data=np.zeros((50000,1,1,3),dtype=np.uint8),targets=labels),False:SimpleNamespace(data=np.zeros((10000,1,1,3),dtype=np.uint8),targets=np.arange(10000)%100)}
        with patch('pprtp.full_data.CIFAR100',side_effect=lambda root,train=True,download=True:fixtures[train]):
            results=[prepare_cifar100('unused',ownership_seed=g)[2] for g in (120100,1,1)]
        self.assertEqual(json.loads(json.dumps(results[0])),old)
        self.assertEqual(results[1],results[2]);new=results[1]
        self.assertNotEqual(new['class_sets_sha256'],old['class_sets_sha256'])
        self.assertNotEqual(new['ownership_order_sha256'],old['ownership_order_sha256'])
        self.assertEqual(new['anchor_indices'],old['anchor_indices'])
        self.assertEqual(new['anchor_indices_sha256'],old['anchor_indices_sha256'])
        self.assertEqual(new['test_indices'],old['test_indices'])
        self.assertEqual(sorted(new['anchor_indices']+sum(new['train_indices'],[])),list(range(50000)))
        self.assertTrue(all(len(cs)==len(set(cs))==20 for cs in new['class_sets']))
        for c,oo in new['owners'].items():
            self.assertEqual(len(oo),2)
            self.assertLessEqual(abs(new['class_counts'][oo[0]][c]-new['class_counts'][oo[1]][c]),1)

    def test_new_graph_three_arm_initialization_and_actual_batch_pairing(self):
        torch.set_num_threads(1);torch.manual_seed(140);sets,_=cifar100_ownership(1)
        local=[TensorDataset(torch.randn(40,3,32,32),torch.tensor(cs*2)) for cs in sets]
        test=TensorDataset(torch.randn(100,3,32,32),torch.arange(100));anchors=TensorDataset(torch.randn(256,3,32,32),torch.zeros(256,dtype=torch.long))
        with tempfile.TemporaryDirectory() as output:
            argv=['pprtp','--data','unused','--output',output,'--device','cpu','--modes','local','fedproto','fedgh','--seeds','0','--rounds','1','--full-data','--dataset','CIFAR100','--num-classes','100','--k','20','--ownership-seed','1']
            with patch('sys.argv',argv),patch('pprtp.full_data.prepare_cifar100',return_value=(local,test,dict(class_sets=sets),anchors)) as prepare,contextlib.redirect_stdout(io.StringIO()):main()
            self.assertEqual(prepare.call_count,3);prepare.assert_called_with('unused',1,2)
            folders=[Path(output)/f'{m}_seed0' for m in ('local','fedproto','fedgh')]
            meta=[json.loads((r/'metadata.json').read_text()) for r in folders]
            self.assertEqual(len({m['initial_state_sha256'] for m in meta}),1)
            self.assertTrue(all(m['ownership_seed']==1 and not m['mixed_backbone'] for m in meta))
            rr=[json.loads((r/'final.json').read_text()) for r in folders]
            self.assertEqual(rr[0]['batch_hashes'],rr[1]['batch_hashes']);self.assertEqual(rr[0]['batch_hashes'],rr[2]['batch_hashes'])
            self.assertEqual(rr[0]['client_model_hashes'],rr[2]['client_model_hashes'])
            self.assertTrue(rr[2]['full_pair_probe']['same_raw_means_counts_exact'])
            self.assertTrue(json.loads((Path(output)/'round_one_pairing_seed0.json').read_text())['passed'])

    def test_nested_one_owner_historical_order_and_coverage(self):
        old=json.loads(Path('research_log/H12A/full/artifacts/experiment/local_seed0/split.json').read_text())
        labels=np.zeros(50000,dtype=np.int64)
        for ii,counts,cs in zip(old['train_indices'],old['class_counts'],old['class_sets']):
            offset=0
            for c in cs:
                n=counts[str(c)];labels[ii[offset:offset+n]]=c;offset+=n
        reserved=[]
        class Train:
            data=np.zeros((50000,1,1,3),dtype=np.uint8)
            @property
            def targets(self):
                assert reserved
                return labels
        test=SimpleNamespace(data=np.zeros((10000,1,1,3),dtype=np.uint8),targets=np.arange(10000)%100)
        def reserve(n):
            a=reserve_anchors(n);reserved.extend(a);return a
        with patch('pprtp.full_data.CIFAR100',side_effect=lambda root,train=True,download=True:Train() if train else test),patch('pprtp.full_data.reserve_anchors',side_effect=reserve):
            default=prepare_cifar100('unused',120100,2)[2]
            local,evaluation,new,anchors=prepare_cifar100('unused',120100,1)
        self.assertEqual(json.dumps(default),json.dumps(old))
        self.assertEqual(new['ownership_order'],old['ownership_order'])
        self.assertEqual(new['anchor_indices'],old['anchor_indices'])
        self.assertEqual(new['test_indices'],old['test_indices'])
        self.assertEqual(sorted(new['anchor_indices']+sum(new['train_indices'],[])),list(range(50000)))
        self.assertEqual([len(cs) for cs in new['class_sets']],[10]*10)
        for j,c in enumerate(old['ownership_order']):
            self.assertEqual(new['owners'][c],[j%10]);self.assertIn(c,old['class_sets'][j%10])
        for i,cs in enumerate(new['class_sets']):
            self.assertEqual(sorted(local[i].tensors[1].unique().tolist()),cs)
            for cs2 in new['class_sets'][i+1:]:self.assertFalse(set(cs)&set(cs2))
        self.assertEqual(len(evaluation),10000);self.assertEqual(new['train_count'],49744)
        self.assertTrue(torch.equal(anchors.tensors[1],torch.zeros(256,dtype=torch.long)))

    def test_one_owner_three_arm_entrypoint(self):
        torch.set_num_threads(1);torch.manual_seed(160);sets,_=cifar100_ownership(120100,1)
        local=[TensorDataset(torch.randn(20,3,32,32),torch.tensor(cs*2)) for cs in sets]
        test=TensorDataset(torch.randn(100,3,32,32),torch.arange(100));anchors=TensorDataset(torch.randn(256,3,32,32),torch.zeros(256,dtype=torch.long))
        with tempfile.TemporaryDirectory() as output:
            argv=['pprtp','--data','unused','--output',output,'--device','cpu','--modes','local','fedproto','fedgh','--seeds','0','--rounds','1','--full-data','--dataset','CIFAR100','--num-classes','100','--k','10','--owners-per-class','1']
            with patch('sys.argv',argv),patch('pprtp.full_data.prepare_cifar100',return_value=(local,test,dict(class_sets=sets),anchors)) as prepare,contextlib.redirect_stdout(io.StringIO()):main()
            prepare.assert_called_with('unused',120100,1)
            f=json.loads((Path(output)/'fedgh_seed0/final.json').read_text())
            self.assertEqual(f['local_optimizer_steps'],[1]*10)
            self.assertIsNone(f['owner_prototype_compatibility'])
            self.assertEqual(f['communication_bytes']['upload_vectors'],100*512*4)
            self.assertTrue(f['full_pair_probe']['same_raw_means_counts_exact'])
            self.assertTrue(json.loads((Path(output)/'round_one_pairing_seed0.json').read_text())['passed'])
