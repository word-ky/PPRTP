import ast,copy,contextlib,io,json,tempfile,time,unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,TensorDataset
from test_baseline import fixture
from pprtp.fedtgp import FedTGPClient,FedTGPServer,TrainableGlobalPrototypes,adaptive_gap,distance_logits
from pprtp.run import main,tensor_hash

def upstream(file):
    tree=ast.parse(Path('vendor/FedTGP/system/flcore/'+file).read_text())
    tree.body=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))]
    ns=dict(torch=torch,nn=nn,F=F,np=np,copy=copy,time=time,defaultdict=defaultdict,DataLoader=DataLoader,Client=object,Server=object)
    exec(compile(tree,file,'exec'),ns);return ns

class FedTGPTest(unittest.TestCase):
    def test_client_matches_pinned_checkpoint_timing_and_observed_mse(self):
        torch.set_num_threads(1);adapted=fixture(FedTGPClient);adapted.lamda=10
        ns=upstream('clients/clienttgp.py');original=object.__new__(ns['clientTGP'])
        original.__dict__.update(adapted.__dict__);original.role='Client';original.loss_mse=nn.MSELoss()
        items={('Client','model'):copy.deepcopy(adapted.model),('Server','global_protos'):None}
        ns['load_item']=lambda role,name,path:copy.deepcopy(items[(role,name)])
        ns['save_item']=lambda item,role,name,path:items.__setitem__((role,name),copy.deepcopy(item))
        for _ in range(2):
            original.train();adapted.train()
            for k,v in items[('Client','model')].state_dict().items():torch.testing.assert_close(v,adapted.model.state_dict()[k],atol=1e-7,rtol=1e-6)
            self.assertEqual(set(adapted.protos),{0,1})
            for k,v in items[('Client','protos')].items():torch.testing.assert_close(v,adapted.protos[k],atol=1e-7,rtol=1e-6)
            bank={k:torch.randn(512) for k in range(10)};items[('Server','global_protos')]=bank;adapted.set_protos(bank)
        self.assertEqual(adapted.optimizer_steps,2)

    def test_server_architecture_gap_objective_and_update_match_official(self):
        torch.set_num_threads(1);ns=upstream('servers/servertgp.py');torch.manual_seed(0)
        official=ns['Trainable_Global_Prototypes'](4,8,8,'cpu');torch.manual_seed(0);ported=TrainableGlobalPrototypes(4,8,'cpu')
        self.assertEqual(tensor_hash(official.state_dict().values()),tensor_hash(ported.state_dict().values()))
        clients=[SimpleNamespace(id=i,role=str(i),save_folder_name='',protos={k:torch.randn(8)+k for k in range(4)}) for i in range(2)]
        server=object.__new__(ns['FedTGP']);server.__dict__.update(selected_clients=clients,num_classes=4,device='cpu',role='Server',save_folder_name='',server_learning_rate=.01,batch_size=8,server_epochs=1,margin_threthold=100,CEloss=nn.CrossEntropyLoss())
        items={(c.role,'protos'):c.protos for c in clients};items[('Server','TGP')]=official
        ns['load_item']=lambda role,name,path:items[(role,name)]
        ns['save_item']=lambda item,role,name,path:items.__setitem__((role,name),item)
        with contextlib.redirect_stdout(io.StringIO()):server.receive_protos()
        torch.testing.assert_close(adaptive_gap(clients,4,'cpu'),server.gap)
        actual=FedTGPServer(4,8,'cpu',batch_size=8,epochs=1)
        with contextlib.redirect_stdout(io.StringIO()):server.update_TGP()
        bank,receipt=actual.update(clients,tensor_hash)
        for k,v in official.state_dict().items():torch.testing.assert_close(v,actual.model.state_dict()[k],atol=1e-6,rtol=1e-5)
        self.assertEqual(receipt['optimizer_steps'],1);self.assertEqual(set(bank),set(range(4)))

    def test_fedtgp_entrypoint_no_transport_and_allclass_readout(self):
        torch.set_num_threads(1);torch.manual_seed(150)
        split=json.loads(Path('research_log/H12A/full/artifacts/experiment/local_seed0/split.json').read_text())
        local=[TensorDataset(torch.randn(40,3,32,32),torch.tensor(cs*2)) for cs in split['class_sets']]
        test=TensorDataset(torch.randn(100,3,32,32),torch.arange(100));anchors=TensorDataset(torch.full((256,3,32,32),float('nan')),torch.full((256,),-999))
        init=FedTGPServer.__init__
        def short(self,*args,**kw):kw['epochs']=2;init(self,*args,**kw)
        with tempfile.TemporaryDirectory() as output:
            argv=['pprtp','--data','unused','--output',output,'--device','cpu','--modes','fedtgp','--seeds','0','--rounds','2','--full-data','--dataset','CIFAR100','--num-classes','100','--k','20']
            with patch('sys.argv',argv),patch('pprtp.full_data.prepare_cifar100',return_value=(local,test,split,anchors)),patch.object(FedTGPServer,'__init__',short),patch('pprtp.full_data.full_readouts',side_effect=AssertionError('transport called')),patch('pprtp.direct_prototypes.analyze_direct',side_effect=AssertionError('transport called')),contextlib.redirect_stdout(io.StringIO()):main()
            path=Path(output)/'fedtgp_seed0';rr=[json.loads(x) for x in (path/'rounds.jsonl').read_text().splitlines()]
            self.assertEqual(len(rr),2);self.assertEqual(rr[-1]['fedtgp_server']['total_optimizer_steps'],28)
            self.assertEqual(rr[-1]['local_optimizer_steps'],[2]*10)
            self.assertEqual(len(rr[-1]['prototype_norms']),100)
            self.assertEqual(len(rr[-1]['prediction_histograms']['l2']),100)
            self.assertFalse(rr[-1]['fedtgp_server']['anchors_used']);self.assertFalse(rr[-1]['fedtgp_server']['test_data_used'])
            for counts,cs in zip(rr[-1]['uploaded_class_counts'],split['class_sets']):self.assertEqual(set(map(int,counts)),set(cs))
