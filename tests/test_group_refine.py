import inspect,unittest
from types import SimpleNamespace
import torch
from torch.utils.data import TensorDataset
from pprtp.group_refine import refine,evaluate_refine
from pprtp.direct_prototypes import analyze_direct,cosine_scores
from pprtp.dual_space import analyze_dual
from pprtp.run import tensor_hash,metrics

class GroupRefineTest(unittest.TestCase):
    def test_routing_and_arbitrary_missing_correctness(self):
        z=torch.eye(10);bank=z.clone();labels=torch.tensor([1,4]);raw=z[labels.flip(0)]
        t=(torch.zeros(10),torch.eye(10),torch.zeros(10))
        pred,shared,route=refine(z,raw,labels,bank,t)
        self.assertEqual(shared.tolist(),list(range(10)))
        self.assertTrue(torch.equal(pred[~route],shared[~route]))
        self.assertTrue(torch.equal(pred[route],labels[cosine_scores(z,raw).argmax(1)][route]))
        self.assertEqual(pred[route].tolist(),[4,1])
        self.assertEqual(list(inspect.signature(refine).parameters),['z','raw','owner_labels','bank','t'])
        for y in range(10):
            if y not in labels:self.assertTrue(torch.equal(pred==y,shared==y))

    def test_isolation_label_perturbation_and_reference(self):
        torch.manual_seed(110);clients=[];local=[]
        for i in range(10):
            model=torch.nn.Module();model.base=torch.nn.Identity();model.head=torch.nn.Linear(10,10);model.head.eval()
            for p in model.parameters():p.grad=torch.ones_like(p)
            classes=sorted([i,(i+1)%10]);y=torch.tensor(classes).repeat_interleave(4)
            local.append(TensorDataset(torch.eye(10)[y]+.02*torch.randn(8,10),y))
            clients.append(SimpleNamespace(model=model,device='cpu',class_set=classes,protos={classes[0]:torch.ones(10)}))
        anchors=TensorDataset(torch.randn(25,10),torch.zeros(25,dtype=torch.long));test=TensorDataset(torch.eye(10),torch.arange(10))
        capture={};args=(clients,clients[0].model.head,anchors,local,test,tensor_hash,metrics,None)
        old=analyze_direct(*args,construction_output=capture)
        dual=analyze_dual(clients,clients[0].model.head,test,capture,tensor_hash,metrics)
        a=evaluate_refine(clients,clients[0].model.head,test,capture,tensor_hash,metrics)
        b=evaluate_refine(clients,clients[0].model.head,TensorDataset(test.tensors[0],test.tensors[1].roll(1)),capture,tensor_hash,metrics)
        self.assertEqual(a['prediction_hashes'],b['prediction_hashes']);self.assertEqual(a['route_hashes'],b['route_hashes'])
        self.assertTrue(a['isolation']['state_rng_modes_gradients_unchanged'])
        self.assertEqual(a['shared_per_client'],old['per_client'])
        self.assertEqual(analyze_direct(*args),old)
        self.assertEqual(analyze_dual(clients,clients[0].model.head,test,capture,tensor_hash,metrics),dual)
