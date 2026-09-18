import unittest
from types import SimpleNamespace
import torch
from torch.utils.data import TensorDataset
from pprtp.direct_prototypes import global_means,cosine_scores,analyze_direct
from pprtp.class_prototypes import class_means,analyze_prototypes
from pprtp.run import tensor_hash,metrics


class DirectPrototypeTest(unittest.TestCase):
    def test_hierarchical_weighting_sparse_classes_and_scale(self):
        torch.manual_seed(51)
        xs=[torch.randn(7,5),torch.randn(9,5)]
        ys=[torch.tensor([2,2,2,2,7,7,7]),torch.tensor([7,7,2,2,2,2,2,2,2])]
        local=[class_means(x,y) for x,y in zip(xs,ys)]
        bank,classes=global_means(torch.cat([a[0] for a in local]),torch.cat([a[1] for a in local]),torch.cat([a[2] for a in local]))
        self.assertEqual(classes.tolist(),[2,7])
        allx=torch.cat(xs);ally=torch.cat(ys)
        for j,c in enumerate(classes): torch.testing.assert_close(bank[j],allx[ally==c].mean(0),atol=1e-7,rtol=1e-6)
        logits=cosine_scores(allx,bank)
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.equal(classes[logits.argmax(1)],classes[(logits*17).argmax(1)]))
        self.assertTrue(set(classes[logits.argmax(1)].tolist())<={2,7})
        self.assertTrue(torch.isfinite(cosine_scores(torch.zeros(2,5),bank)).all())

    def test_direct_path_state_and_rng(self):
        torch.manual_seed(52);clients=[];support=[]
        for i in range(2):
            model=torch.nn.Module();model.base=torch.nn.Identity();model.head=torch.nn.Linear(10,10)
            model.head.eval()
            for p in model.parameters(): p.grad=torch.ones_like(p)
            classes=list(range(i*5,(i+1)*5));y=torch.tensor(classes).repeat_interleave(2)
            support.append(TensorDataset(torch.eye(10)[y],y))
            clients.append(SimpleNamespace(model=model,device='cpu',class_set=classes,protos={classes[0]:torch.ones(10)}))
        anchors=TensorDataset(torch.randn(25,10),torch.zeros(25,dtype=torch.long));test=TensorDataset(torch.eye(10),torch.arange(10))
        rng=torch.get_rng_state().clone();args=(clients,clients[0].model.head,anchors,support,test,tensor_hash,metrics)
        reference=analyze_prototypes(*args)
        for aligned in (True,False):
            a=analyze_direct(*args,reference,aligned=aligned)
            self.assertEqual(a['global_labels'],list(range(10)))
            self.assertEqual(a['state_before'],a['state_after']);self.assertFalse(a['fitting'])
            self.assertEqual(sum(a['prediction_histograms']['total']['overall']),20)
            for key in ('rng_cpu_unchanged','rng_cuda_unchanged','module_modes_unchanged','existing_gradients_unchanged'):
                self.assertTrue(a[key])
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
