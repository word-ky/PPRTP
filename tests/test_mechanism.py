import unittest
from types import SimpleNamespace
import pprtp
import torch
from test_baseline import fixture
from pprtp.client import H01Client, knowledge_loss, aggregate, prototype_bank
from pprtp.run import tensor_hash, check_round_one


class MechanismTests(unittest.TestCase):
    def test_round_one_pairing_all_modes_and_grad_ratio(self):
        records=[]
        for mode in ('local','fedproto','gpc'):
            client=fixture(H01Client)
            client.mode=mode
            client.train()
            records.append(dict(client_model_hashes=[tensor_hash(client.model.state_dict().values())],
                prototype_bank_hash=tensor_hash([client.protos[k] for k in sorted(client.protos)])))
            self.assertEqual(client.diagnostic['scaled_knowledge_grad_norm'],0.)
            self.assertGreater(client.diagnostic['local_grad_norm'],0.)
            client.set_protos(client.protos)
            client.train()
            d=client.diagnostic
            self.assertAlmostEqual(d['scaled_knowledge_grad_norm']/d['local_grad_norm'],
                                   d['knowledge_local_grad_ratio'],places=6)
        check_round_one(records)
        records[-1]['prototype_bank_hash']='different'
        with self.assertRaises(AssertionError):
            check_round_one(records)

    def test_missing_class_changes_gpc_loss_and_gradient_only(self):
        torch.manual_seed(7)
        z = torch.randn(4, 5, requires_grad=True)
        y = torch.tensor([0, 1, 0, 1])
        bank = torch.randn(10, 5, requires_grad=True)
        changed = bank.detach().clone()
        changed[8] = z[0].detach()
        valid = torch.ones(10, dtype=torch.bool)
        for mode in ("fedproto", "gpc"):
            a = knowledge_loss(z, y, bank, valid, mode)
            b = knowledge_loss(z, y, changed, valid, mode)
            ga = torch.autograd.grad(a, z, retain_graph=True)[0]
            gb = torch.autograd.grad(b, z, retain_graph=True)[0]
            if mode == "gpc":
                self.assertGreater(abs(a.item()-b.item()), 1e-5)
                self.assertGreater((ga-gb).abs().max().item(), 1e-5)
            else:
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                torch.testing.assert_close(ga, gb, rtol=0, atol=0)
            self.assertIsNone(torch.autograd.grad(a, bank, allow_unused=True)[0])

    def test_validity_and_noncontiguous_labels(self):
        z = torch.tensor([[1., 0.], [0., 1.]], requires_grad=True)
        bank, valid = prototype_bank({2: z[0], 8: z[1]}, 10, z)
        y = torch.tensor([2, 8])
        actual = knowledge_loss(z, y, bank, valid, "gpc", 1.)
        expected = torch.nn.functional.cross_entropy(torch.eye(2), torch.tensor([0, 1]))
        torch.testing.assert_close(actual, expected)
        valid[:] = False
        zero = knowledge_loss(z, y, bank, valid, "gpc")
        self.assertEqual(zero.item(), 0)
        self.assertTrue(torch.isfinite(torch.autograd.grad(zero, z)[0]).all())

    def test_sample_weighted_aggregation(self):
        clients = [SimpleNamespace(protos={0: torch.tensor([1., 2.])}, proto_counts={0: 1}),
                   SimpleNamespace(protos={0: torch.tensor([5., 6.]), 3: torch.tensor([2., 3.])},
                                   proto_counts={0: 3, 3: 2})]
        result = aggregate(clients)
        torch.testing.assert_close(result[0], torch.tensor([4., 5.]))
        self.assertEqual(set(result), {0, 3})

    def test_training_parity_with_unmodified_upstream(self):
        original, adapted = fixture(), fixture(H01Client)
        for _ in range(2):
            original.train()
            adapted.train()
            for key, value in original.model.state_dict().items():
                torch.testing.assert_close(value, adapted.model.state_dict()[key], atol=1e-7, rtol=1e-6)
            for c, p in original.protos.items():
                torch.testing.assert_close(p, adapted.protos[c], atol=1e-7, rtol=1e-6)
            original.set_protos(original.protos)
            adapted.set_protos(original.global_protos)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
