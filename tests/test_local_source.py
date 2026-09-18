import unittest
from types import SimpleNamespace
import torch
from torch.utils.data import TensorDataset
from pprtp.direct_prototypes import analyze_direct
from pprtp.local_source import local_provenance,source_shift,index_hash
from pprtp.run import tensor_hash,metrics


class LocalSourceTest(unittest.TestCase):
    def test_exact_local_inputs_hierarchy_and_isolation(self):
        torch.manual_seed(53);clients=[];local=[];indices=[];sets=[]
        for i in range(10):
            classes=sorted([i,(i+1)%10]);sets.append(classes)
            y=torch.tensor(classes).repeat_interleave(100)
            local.append(TensorDataset(torch.eye(10)[y]+.1,y));indices.append(list(range(i*200,(i+1)*200)))
            model=torch.nn.Module();model.base=torch.nn.Identity();model.head=torch.nn.Linear(10,10);model.head.eval()
            for p in model.parameters(): p.grad=torch.ones_like(p)
            clients.append(SimpleNamespace(model=model,device='cpu',class_set=classes,protos={classes[0]:torch.full((10,),999.)}))
        split=dict(train_indices=indices,class_sets=sets,official_train_test_separate=True)
        heldout_indices=[list(range(10000+i*200,10200+i*200)) for i in range(10)]
        receipt=local_provenance(local,split,[30000],heldout_indices,list(range(20000,20025)),tensor_hash)
        self.assertEqual(receipt['indices_sha256'],index_hash(indices));self.assertEqual(receipt['refresh_forward_examples_total'],2000)
        self.assertEqual(receipt['per_client_index_sha256'],[index_hash(ii) for ii in indices])
        anchors=TensorDataset(torch.randn(25,10),torch.zeros(25,dtype=torch.long));test=TensorDataset(torch.eye(10),torch.arange(10))
        rng=torch.get_rng_state().clone();bank={}
        a=analyze_direct(clients,clients[0].model.head,anchors,local,test,tensor_hash,metrics,None,bank_output=bank)
        for p in a['local_prototypes']:
            ds=local[p['client']];expected=ds.tensors[0][ds.tensors[1]==p['label']].mean(0)
            self.assertEqual(p['raw_hash'],tensor_hash([expected]));self.assertEqual(p['count'],100)
        b=analyze_direct(clients,clients[0].model.head,anchors,local,test,tensor_hash,metrics,None,aligned=False)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for arm in (a,b):
            self.assertEqual(arm['state_before'],arm['state_after'])
            for key in ('rng_cpu_unchanged','rng_cuda_unchanged','module_modes_unchanged','existing_gradients_unchanged'):
                self.assertTrue(arm[key])
            self.assertLess(max(p['hierarchical_max_error'] for p in arm['global_prototypes']),1e-6)
        self.assertEqual([s['l2'] for s in source_shift(bank['bank'],bank['bank'])],[0.]*10)
