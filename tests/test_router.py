import inspect,unittest
from types import SimpleNamespace
import torch
from torch.utils.data import TensorDataset
from pprtp.router import loo_means,loo_radius,route,calibrate_radii,evaluate_router
from pprtp.direct_prototypes import analyze_direct
from pprtp.run import tensor_hash,metrics


class RouterTest(unittest.TestCase):
    def test_loo_and_order_statistic(self):
        torch.manual_seed(101);z=torch.randn(100,7);p=z.mean(0)
        brute=torch.stack([torch.cat([z[:j],z[j+1:]]).mean(0) for j in range(100)])
        torch.testing.assert_close(loo_means(z,p),brute,atol=1e-7,rtol=1e-5)
        q,a,rank=loo_radius(z,p);self.assertEqual(rank,91);self.assertTrue(torch.equal(q,a.sort().values[90]))
        self.assertTrue(torch.equal(q,loo_radius(z,p)[0]))

    def test_eligible_prediction_and_rejection(self):
        z=torch.tensor([[1.,0.],[-1.,0.]])
        raw=torch.tensor([[.8,.6],[.98,.199]]);labels=torch.tensor([0,2]);bank=torch.tensor([[0.,1.]]*10);bank[4]=torch.tensor([-1.,0.])
        t=(torch.zeros(2),torch.eye(2),torch.zeros(2));q=torch.tensor([.3,.001])
        pred,accepted,_,_=route(z,raw,labels,bank,t,q)
        self.assertEqual(pred.tolist(),[0,4]);self.assertEqual(accepted.tolist(),[True,False])
        self.assertNotIn('y',inspect.signature(route).parameters);self.assertNotIn('test',inspect.signature(calibrate_radii).parameters)

    def test_train_only_isolation_and_label_perturbation(self):
        torch.manual_seed(102);clients=[];local=[]
        for i in range(10):
            model=torch.nn.Module();model.base=torch.nn.Identity();model.head=torch.nn.Linear(10,10);model.head.eval()
            for p in model.parameters(): p.grad=torch.ones_like(p)
            classes=sorted([i,(i+1)%10]);y=torch.tensor(classes).repeat_interleave(100)
            local.append(TensorDataset(torch.eye(10)[y]+.02*torch.randn(200,10),y))
            clients.append(SimpleNamespace(model=model,device='cpu',class_set=classes,protos={classes[0]:torch.ones(10)}))
        anchors=TensorDataset(torch.randn(25,10),torch.zeros(25,dtype=torch.long));test=TensorDataset(torch.eye(10),torch.arange(10))
        capture={};args=(clients,clients[0].model.head,anchors,local,test,tensor_hash,metrics,None)
        old=analyze_direct(*args,construction_output=capture)
        q,receipt=calibrate_radii(clients,clients[0].model.head,local,capture,tensor_hash)
        self.assertEqual(len(receipt['radii']),20);self.assertTrue(receipt['isolation']['state_rng_modes_gradients_unchanged'])
        a=evaluate_router(clients,clients[0].model.head,test,capture,q,tensor_hash,metrics)
        b=evaluate_router(clients,clients[0].model.head,TensorDataset(test.tensors[0],test.tensors[1].roll(1)),capture,q,tensor_hash,metrics)
        self.assertEqual(a['prediction_histograms']['total']['overall'],b['prediction_histograms']['total']['overall'])
        self.assertEqual(a['routing_diagnostics']['total']['native_count'],b['routing_diagnostics']['total']['native_count'])
        self.assertTrue(a['isolation']['state_rng_modes_gradients_unchanged'])
        self.assertEqual(q[0].numel()*q[0].element_size(),8)
        self.assertEqual(analyze_direct(*args),old)
