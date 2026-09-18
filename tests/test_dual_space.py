import unittest
from types import SimpleNamespace
import torch
from torch.utils.data import TensorDataset
from pprtp.direct_prototypes import analyze_direct,cosine_scores
from pprtp.dual_space import dual_scores,analyze_dual
from pprtp.paired import transform
from pprtp.run import tensor_hash,metrics


class DualSpaceTest(unittest.TestCase):
    def test_exact_score_columns(self):
        torch.manual_seed(91);z=torch.randn(7,5);raw=torch.randn(2,5);labels=torch.tensor([2,8]);bank=torch.randn(10,5)
        t=(torch.randn(5),torch.eye(5),torch.randn(5))
        s=dual_scores(z,raw,labels,bank,t);native=cosine_scores(z,raw);aligned=cosine_scores(transform(z,t),bank)
        self.assertTrue(torch.equal(s[:,labels],native))
        missing=[i for i in range(10) if i not in labels]
        self.assertTrue(torch.equal(s[:,missing],aligned[:,missing]))

    def test_capture_reference_isolation_and_label_independence(self):
        torch.manual_seed(92);clients=[];local=[]
        for i in range(2):
            model=torch.nn.Module();model.base=torch.nn.Identity();model.head=torch.nn.Linear(10,10);model.head.eval()
            for p in model.parameters(): p.grad=torch.ones_like(p)
            classes=list(range(i*5,(i+1)*5));y=torch.tensor(classes).repeat_interleave(2)
            local.append(TensorDataset(torch.eye(10)[y],y))
            clients.append(SimpleNamespace(model=model,device='cpu',class_set=classes,protos={classes[0]:torch.ones(10)}))
        anchors=TensorDataset(torch.randn(25,10),torch.zeros(25,dtype=torch.long));test=TensorDataset(torch.eye(10),torch.arange(10))
        args=(clients,clients[0].model.head,anchors,local,test,tensor_hash,metrics,None)
        old=analyze_direct(*args);capture={};actual=analyze_direct(*args,expected_alignment=old['alignment'],construction_output=capture)
        self.assertEqual(old,actual)
        for i,(raw,labels) in enumerate(capture['raw_owners']):
            for p,c in zip(raw,labels.tolist()):
                self.assertEqual(tensor_hash([p]),next(v['raw_hash'] for v in old['local_prototypes'] if v['client']==i and v['label']==c))
        a=analyze_dual(clients,clients[0].model.head,test,capture,tensor_hash,metrics)
        changed=TensorDataset(test.tensors[0],test.tensors[1].roll(1))
        b=analyze_dual(clients,clients[0].model.head,changed,capture,tensor_hash,metrics)
        self.assertEqual(a['prediction_histograms']['total']['overall'],b['prediction_histograms']['total']['overall'])
        self.assertEqual(a['winning_group_fraction'],b['winning_group_fraction'])
        self.assertEqual(a['state_before'],a['state_after']);self.assertTrue(a['state_rng_modes_gradients_unchanged'])
        self.assertEqual(a['metrics']['all'],1.)

        centered=analyze_dual(clients,clients[0].model.head,test,capture,tensor_hash,metrics,centered=True)
        changed_centered=analyze_dual(clients,clients[0].model.head,changed,capture,tensor_hash,metrics,centered=True)
        self.assertEqual(centered['prediction_histograms']['total']['overall'],changed_centered['prediction_histograms']['total']['overall'])
        self.assertTrue(centered['state_rng_modes_gradients_unchanged'])
        self.assertEqual(centered['state_before'],centered['state_after'])
        self.assertLess(centered['owner_rotation_max_abs_error'],2e-5)
        self.assertEqual(analyze_dual(clients,clients[0].model.head,test,capture,tensor_hash,metrics),a)
