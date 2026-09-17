import copy
from types import SimpleNamespace
import unittest
import torch
from pprtp.fedgh import broadcast, train_server, fit_probe


class FedGHTest(unittest.TestCase):
    def test_probe_separable_and_side_effect_free(self):
        torch.manual_seed(7)
        head=torch.nn.Linear(10,10)
        old=copy.deepcopy(head.state_dict())
        x=torch.eye(10,requires_grad=True)
        clients=[SimpleNamespace(protos={i:x[i] for i in range(10)})]
        probe,info=fit_probe(head,clients)
        self.assertLess(info['after']['ce'],info['before']['ce']*.01)
        self.assertGreaterEqual(info['after']['accuracy'],.95)
        self.assertIsNone(x.grad)
        for k,v in head.state_dict().items():
            self.assertTrue(torch.equal(v,old[k]))
        self.assertTrue(all(p.grad is None for p in head.parameters()))
        self.assertTrue(all(torch.isfinite(p).all() for p in probe.parameters()))

    def test_server_update_detach_broadcast_and_all_classes(self):
        torch.manual_seed(4)
        clients=[]
        for i in range(10):
            model=torch.nn.Module()
            model.base=torch.nn.Linear(4,8)
            model.head=torch.nn.Linear(8,10)
            clients.append(SimpleNamespace(id=i,model=model,
                protos={c:model.base(torch.randn(4)) for c in (i,(i+1)%10)}))
        base_states=[copy.deepcopy(c.model.base.state_dict()) for c in clients]
        head=copy.deepcopy(clients[0].model.head)
        old=copy.deepcopy(head.state_dict())
        optimizer=torch.optim.SGD(head.parameters(),lr=.01)
        info=train_server(head,optimizer,clients)
        self.assertEqual(len(info['sample_order']),20)
        self.assertTrue(any(not torch.equal(old[k],v) for k,v in head.state_dict().items()))
        broadcast(head,clients)
        for c,state in zip(clients,base_states):
            for k,v in c.model.base.state_dict().items():
                self.assertTrue(torch.equal(v,state[k]))
            self.assertTrue(all(p.grad is None for p in c.model.base.parameters()))
            self.assertEqual(c.model.head.weight.shape,(10,8))
            for k,v in c.model.head.state_dict().items():
                self.assertTrue(torch.equal(v,head.state_dict()[k]))
            self.assertTrue(torch.isfinite(c.model.head(c.model.base(torch.randn(3,4)))).all())
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in head.parameters()))


if __name__=='__main__':
    unittest.main()
