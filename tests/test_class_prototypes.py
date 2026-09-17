import unittest
from types import SimpleNamespace
import torch
from torch.utils.data import TensorDataset
from pprtp.class_prototypes import class_means,analyze_prototypes
from pprtp.paired import transform
from pprtp.fedgh import weighted_ce
from pprtp.run import tensor_hash,metrics


class ClassPrototypeTest(unittest.TestCase):
    def test_affine_mean_weighting_and_class_isolation(self):
        torch.manual_seed(49)
        z=torch.randn(8,10);y=torch.tensor([2,0,2,0,0,2,2,0])
        p,lab,count=class_means(z,y)
        self.assertEqual(lab.tolist(),[0,2]);self.assertEqual(count.tolist(),[4,4])
        torch.testing.assert_close(p[0],z[y==0].mean(0));torch.testing.assert_close(p[1],z[y==2].mean(0))
        t=(torch.randn(10),torch.randn(10,10),torch.randn(10))
        torch.testing.assert_close(transform(p,t),class_means(transform(z,t),y)[0],atol=1e-6,rtol=1e-5)
        logits=torch.randn(3,10,dtype=torch.double,requires_grad=True);labels=torch.tensor([0,3,9]);counts=torch.tensor([1,3,7])
        weighted=weighted_ce(logits,labels,counts)
        repeated=torch.nn.functional.cross_entropy(logits.repeat_interleave(counts,dim=0),labels.repeat_interleave(counts))
        torch.testing.assert_close(weighted,repeated,atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(torch.autograd.grad(weighted,logits,retain_graph=True)[0],torch.autograd.grad(repeated,logits)[0],atol=1e-12,rtol=1e-12)

    def test_client_isolation_and_diagnostic_rng(self):
        torch.manual_seed(50);clients=[];support=[]
        for i in range(2):
            model=torch.nn.Module();model.base=torch.nn.Identity();model.head=torch.nn.Linear(10,10)
            model.head.eval()
            for p in model.parameters(): p.grad=torch.ones_like(p)
            classes=list(range(i*5,(i+1)*5));y=torch.tensor(classes).repeat_interleave(2)
            support.append(TensorDataset(torch.eye(10)[y]+i*.1,y))
            clients.append(SimpleNamespace(model=model,device='cpu',class_set=classes,protos={classes[0]:torch.ones(10)}))
        anchors=TensorDataset(torch.randn(25,10),torch.zeros(25,dtype=torch.long));test=TensorDataset(torch.eye(10),torch.arange(10))
        rng=torch.get_rng_state().clone()
        a=analyze_prototypes(clients,clients[0].model.head,anchors,support,test,tensor_hash,metrics)
        b=analyze_prototypes(clients,clients[0].model.head,anchors,support,test,tensor_hash,metrics,aligned=False)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(a['prototype_count'],10)
        self.assertEqual([(p['client'],p['label'],p['count'],p['raw_hash']) for p in a['prototypes']],[(p['client'],p['label'],p['count'],p['raw_hash']) for p in b['prototypes']])
        for p in a['prototypes']:
            self.assertIn(p['label'],clients[p['client']].class_set);self.assertEqual(p['count'],2)
        for arm in (a,b):
            self.assertEqual(arm['state_before'],arm['state_after'])
            for key in ('rng_cpu_unchanged','rng_cuda_unchanged','module_modes_unchanged','existing_gradients_unchanged'):
                self.assertTrue(arm[key])
