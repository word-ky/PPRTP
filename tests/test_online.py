import inspect,unittest
from types import SimpleNamespace
import torch
from torch.utils.data import TensorDataset
from pprtp.online import build_bank,aligned_loss
from pprtp.run import tensor_hash


class OnlineTest(unittest.TestCase):
    def test_pure_bank_and_isolation(self):
        torch.manual_seed(81);clients=[];datasets=[]
        for i in range(10):
            model=torch.nn.Module();model.base=torch.nn.Linear(10,10);model.head=torch.nn.Linear(10,10);model.head.eval()
            for p in model.parameters(): p.grad=torch.ones_like(p)
            classes=sorted([i,(i+1)%10]);y=torch.tensor(classes).repeat_interleave(100)
            datasets.append(TensorDataset(torch.eye(10)[y]+.1,y))
            clients.append(SimpleNamespace(model=model,device='cpu',class_set=classes))
        anchors=TensorDataset(torch.randn(25,10),torch.zeros(25,dtype=torch.long))
        self.assertNotIn('test',inspect.signature(build_bank).parameters)
        bank,tt,receipt=build_bank(clients,anchors,datasets,tensor_hash)
        again,_,other=build_bank(clients,anchors,datasets,tensor_hash)
        self.assertEqual(receipt,other);self.assertTrue(torch.equal(bank,again))
        self.assertTrue(receipt['state_rng_modes_gradients_unchanged']);self.assertEqual(receipt['semantic_forward_examples'],2000)
        self.assertFalse(bank.requires_grad);self.assertTrue(all(not v.requires_grad for t in tt for v in t))

    def test_mask_and_frozen_backprop(self):
        torch.manual_seed(13);z=torch.randn(4,5,requires_grad=True);y=torch.tensor([2,8,2,8]);seen=[2,8]
        bank=torch.randn(10,5,requires_grad=True)
        t=(torch.randn(5,requires_grad=True),torch.eye(5,requires_grad=True),torch.randn(5,requires_grad=True))
        changed=bank.detach().clone();changed[4]=z[0].detach()
        for all_classes in (True,False):
            a=aligned_loss(z,y,bank,t,seen,all_classes);b=aligned_loss(z,y,changed,t,seen,all_classes)
            ga=torch.autograd.grad(a,z,retain_graph=True)[0];gb=torch.autograd.grad(b,z,retain_graph=True)[0]
            if all_classes:
                self.assertGreater(abs(a.item()-b.item()),1e-5);self.assertGreater((ga-gb).abs().max().item(),1e-5)
            else:
                self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(ga,gb))
            a.backward(retain_graph=True)
            self.assertIsNone(bank.grad);self.assertTrue(all(v.grad is None for v in t))

    def test_round_one_pairing_and_batches(self):
        from test_baseline import fixture
        from pprtp.client import H01Client
        from torch.utils.data import DataLoader
        hashes=[];batches=[]
        for mode in ('fedgh','pprtp_all_lag1','pprtp_seen_lag1'):
            c=fixture(H01Client);c.mode=mode;c.lamda=.002
            ds=c.load_train_data().dataset
            c.load_train_data=lambda: DataLoader(ds,batch_size=4,shuffle=True,generator=torch.Generator().manual_seed(0))
            c.train();hashes.append(tensor_hash(c.model.state_dict().values()))
            if mode!='fedgh': batches.append(c.batch_hashes)
        self.assertEqual(len(set(hashes)),1);self.assertEqual(batches[0],batches[1])
