import contextlib,copy,io,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch.utils.data import DataLoader,TensorDataset
from test_baseline import fixture
from pprtp.client import H01Client
from pprtp.fedavg import MatchedFedAvg
from pprtp.full_data import cifar100_ownership
from pprtp.run import main,tensor_hash
from flcore.clients.clientavg import clientAVG


class FedAvgTest(unittest.TestCase):
    def test_local_sgd_matches_upstream_and_weighted_parameter_broadcast(self):
        torch.set_num_threads(1)
        original=fixture(clientAVG);adapted=fixture(H01Client);adapted.mode='local'
        for _ in range(2):
            original.train();adapted.train()
            self.assertEqual(tensor_hash(original.model.state_dict().values()),tensor_hash(adapted.model.state_dict().values()))
        second=fixture(H01Client);second.mode='local';second.train_samples=16
        for p in second.model.parameters():p.data.add_(.25)
        before=[tensor_hash(c.model.state_dict().values()) for c in (adapted,second)]
        expected=[a.detach().clone()/3+b.detach().clone()* (2/3) for a,b in zip(adapted.model.parameters(),second.model.parameters())]
        server=MatchedFedAvg([adapted,second])
        for actual,wanted in zip(server.global_model.parameters(),expected):torch.testing.assert_close(actual,wanted)
        self.assertEqual(before,[tensor_hash(c.model.state_dict().values()) for c in (adapted,second)])
        for c in (adapted,second):
            c.set_parameters(server.global_model)
            self.assertEqual(tensor_hash(c.model.state_dict().values()),tensor_hash(server.global_model.state_dict().values()))
        original.set_parameters(server.global_model);original.train();adapted.train()
        self.assertEqual(tensor_hash(original.model.state_dict().values()),tensor_hash(adapted.model.state_dict().values()))

    def test_one_owner_four_arm_pairing_and_global_readout(self):
        torch.set_num_threads(1);torch.manual_seed(161);sets,_=cifar100_ownership(120100,1)
        local=[TensorDataset(torch.randn(20,3,32,32),torch.tensor(cs*2)) for cs in sets]
        test=TensorDataset(torch.randn(100,3,32,32),torch.arange(100));anchors=TensorDataset(torch.randn(256,3,32,32),torch.zeros(256,dtype=torch.long))
        with tempfile.TemporaryDirectory() as output:
            argv=['pprtp','--data','unused','--output',output,'--device','cpu','--modes','local','fedproto','fedgh','fedavg','--seeds','0','--rounds','2','--full-data','--dataset','CIFAR100','--num-classes','100','--k','10','--owners-per-class','1']
            with patch('sys.argv',argv),patch('pprtp.full_data.prepare_cifar100',return_value=(local,test,dict(class_sets=sets),anchors)),contextlib.redirect_stdout(io.StringIO()):main()
            path=Path(output)/'fedavg_seed0';rr=[json.loads(x) for x in (path/'rounds.jsonl').read_text().splitlines()]
            self.assertTrue(json.loads((Path(output)/'round_one_pairing_seed0.json').read_text())['passed'])
            self.assertTrue((path/'global_model.pt').exists())
            for r in rr:
                self.assertEqual(r['fedavg_server']['weights'],[.1]*10)
                self.assertEqual(r['local_optimizer_steps'],[1]*10)
                h=[c['global_model_post_server']['prediction_histogram'] for c in r['per_client']]
                self.assertTrue(all(v==h[0] for v in h))
                self.assertEqual(r['losses'][0]['knowledge'],0)
                self.assertGreater(r['communication_bytes']['upload_model_all_clients'],0)
            self.assertNotEqual(rr[0]['fedavg_server']['global_model_hash'],rr[1]['fedavg_server']['global_model_hash'])
