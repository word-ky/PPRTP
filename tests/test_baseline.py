import copy
from types import SimpleNamespace
import unittest

import pprtp
import torch
from torch.utils.data import DataLoader, TensorDataset
from flcore.clients.clientproto import clientProto
from flcore.servers.serverproto import proto_aggregation
from flcore.trainmodel.models import FedAvgCNN, BaseHeadSplit


def make_args(device="cpu"):
    torch.manual_seed(17)
    cnn = FedAvgCNN(in_features=3, num_classes=10, dim=1600)
    head = copy.deepcopy(cnn.fc)
    cnn.fc = torch.nn.Identity()
    return SimpleNamespace(model=BaseHeadSplit(cnn, head).to(device),
        algorithm="FedProto", dataset="Cifar10", device=device,
        save_folder_name="items", num_classes=10, batch_size=4,
        local_learning_rate=0.01, local_epochs=1, few_shot=0,
        learning_rate_decay_gamma=1., learning_rate_decay=False, lamda=1.)


def fixture(client_class=clientProto, device="cpu"):
    args = make_args(device)
    client = client_class(args, 0, 8, 8, train_slow=False, send_slow=False)
    client.class_set = [0, 1]
    torch.manual_seed(29)
    ds = TensorDataset(torch.randn(8, 3, 32, 32), torch.tensor([0, 1]*4))
    client.load_train_data = lambda: DataLoader(ds, batch_size=4, shuffle=False)
    client.load_test_data = client.load_train_data
    return client


class BaselineTest(unittest.TestCase):
    def test_unmodified_upstream_two_rounds(self):
        client = fixture()
        before = copy.deepcopy(client.model.state_dict())
        client.train()
        self.assertEqual(set(client.protos), {0, 1})
        self.assertTrue(any(not torch.equal(v, before[k]) for k,v in client.model.state_dict().items()))
        client.set_protos(proto_aggregation([client.protos]))
        client.train()
        acc, n, _ = client.test_metrics()
        self.assertEqual(n, 8)
        self.assertTrue(0 <= acc <= n)
        self.assertTrue(all(torch.isfinite(p).all() for p in client.protos.values()))


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
