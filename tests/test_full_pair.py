import unittest
from types import SimpleNamespace
import torch
from torch.utils.data import TensorDataset
from pprtp.full_data import full_readouts
from pprtp.direct_prototypes import analyze_direct
from pprtp.paired import break_pairs
from pprtp.run import tensor_hash,metrics

class FullPairTest(unittest.TestCase):
    def test_legacy_permutations_exact_multisets_and_isolation(self):
        torch.manual_seed(113);clients=[];local=[]
        for i in range(10):
            model=torch.nn.Module();model.base=torch.nn.Identity();model.head=torch.nn.Linear(10,10);model.head.eval()
            for p in model.parameters():p.grad=torch.ones_like(p)
            cs=sorted([i,(i+1)%10]);y=torch.tensor(cs).repeat_interleave(4)
            local.append(TensorDataset(torch.eye(10)[y]+.02*torch.randn(8,10),y))
            clients.append(SimpleNamespace(model=model,device='cpu',class_set=cs,protos={cs[0]:torch.ones(10)}))
        x=torch.randn(256,10);anchors=TensorDataset(x,torch.zeros(256,dtype=torch.long));test=TensorDataset(torch.eye(10),torch.arange(10))
        args=(clients,clients[0].model.head,anchors,local,test,tensor_hash,metrics)
        capture={};historical=full_readouts(*args);paired=full_readouts(*args,construction_output=capture)
        self.assertEqual(paired,historical)
        broken=analyze_direct(*args,None,broken=True)
        self.assertEqual(broken['anchor_feature_hashes'],capture['anchor_feature_hashes'])
        self.assertEqual(broken['alignment'][0],paired['pprtp_h07']['alignment'][0])
        for r in broken['permutation_receipts']:
            permuted,legacy=break_pairs(x,r['client'])
            for k,value in legacy.items():self.assertEqual(r[k],value)
            self.assertEqual(r['original_feature_hash'],tensor_hash([x]));self.assertEqual(r['permuted_feature_hash'],tensor_hash([permuted]))
            self.assertEqual(sorted(r['permutation']),list(range(256)))
            self.assertNotEqual(r['permutation'],list(range(256)))
            self.assertTrue(r['multiset_bitwise_unchanged'])
        self.assertEqual([r['fixed_points'] for r in broken['permutation_receipts']],[1,0,1,2,1,0,2,3,1])
        signature=lambda a:[(p['client'],p['label'],p['count'],p['raw_hash']) for p in a['local_prototypes']]
        self.assertEqual(signature(broken),signature(paired['pprtp_h07']))
        self.assertEqual(signature(broken),signature(paired['native_global_prototype_cosine_control']))
        b=analyze_direct(clients,clients[0].model.head,TensorDataset(x,torch.randint(10,(256,))),local,test,tensor_hash,metrics,None,broken=True)
        self.assertEqual(b,broken)
        changed=analyze_direct(clients,clients[0].model.head,anchors,local,TensorDataset(test.tensors[0],test.tensors[1].roll(1)),tensor_hash,metrics,None,broken=True)
        self.assertEqual(broken['alignment'],changed['alignment']);self.assertEqual(broken['global_hash'],changed['global_hash'])
        self.assertEqual(broken['prediction_histograms']['total']['overall'],changed['prediction_histograms']['total']['overall'])
        self.assertEqual(broken['state_before'],broken['state_after']);self.assertTrue(broken['rng_cpu_unchanged'] and broken['module_modes_unchanged'] and broken['existing_gradients_unchanged'])
        self.assertEqual(full_readouts(*args),historical)
