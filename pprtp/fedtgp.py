"""Matched-protocol port of TsingZ0/FedTGP@c77cbbb (Apache-2.0).
See PROVENANCE.md for round-start checkpoint collection semantics.
"""
import copy,time
from collections import defaultdict
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from pprtp.client import H01Client
from flcore.clients.clientproto import agg_func

UPSTREAM_SHA='c77cbbb31eb30d13066cd11f7f4a2e732aeaae24'

class TrainableGlobalPrototypes(nn.Module):
    def __init__(self,num_classes,feature_dim,device):
        super().__init__();self.device=device
        self.embedings=nn.Embedding(num_classes,feature_dim)
        self.middle=nn.Sequential(nn.Sequential(nn.Linear(feature_dim,feature_dim),nn.ReLU()))
        self.fc=nn.Linear(feature_dim,feature_dim)
    def forward(self,class_id):
        return self.fc(self.middle(self.embedings(torch.as_tensor(class_id,device=self.device))))

class FedTGPClient(H01Client):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.mode='fedproto';self.fedtgp=True
        self.prototype_timing=getattr(args[0],'fedtgp_prototype_timing','round_start')
    def train(self):
        # Official collect_protos reloads disk BEFORE train saves the updated model.
        snapshot=copy.deepcopy(self.model) if self.prototype_timing=='round_start' else None
        super().train() # Existing CE + observed-class MSE and matched SGD/batches.
        if snapshot is None:snapshot=self.model
        snapshot.eval()
        protos=defaultdict(list)
        loader=DataLoader(self.load_train_data().dataset,batch_size=self.batch_size,shuffle=False,drop_last=False,generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            for x,y in loader:
                z=snapshot.base(x.to(self.device))
                for j,label in enumerate(y.tolist()):protos[label].append(z[j].detach())
        self.proto_counts={k:len(v) for k,v in protos.items()};self.protos=agg_func(protos)
        assert set(self.protos)==set(self.class_set)

def adaptive_gap(clients,num_classes,device):
    groups=defaultdict(list)
    for c in clients:
        for label,p in c.protos.items():groups[label].append(p.detach())
    assert set(groups)==set(range(num_classes))
    means=torch.stack([torch.stack(groups[k]).mean(0) for k in range(num_classes)]).to(device)
    distances=torch.linalg.vector_norm(means[:,None]-means[None,:],dim=2)
    distances.fill_diagonal_(torch.inf)
    return distances.min(1).values

def distance_logits(proto,generated,y,margin):
    dist=(proto.square().sum(1,keepdim=True)-2*proto@generated.T+generated.square().sum(1)[None]).sqrt()
    return -(dist+F.one_hot(y,len(generated))*margin)

class FedTGPServer:
    def __init__(self,num_classes,feature_dim,device,seed=0,lr=.01,batch_size=32,epochs=100,threshold=100):
        self.num_classes=num_classes;self.device=device;self.lr=lr;self.batch_size=batch_size;self.epochs=epochs;self.threshold=threshold
        with torch.random.fork_rng(devices=[torch.device(device).index or 0] if str(device).startswith('cuda') else []):
            torch.manual_seed(seed);self.model=TrainableGlobalPrototypes(num_classes,feature_dim,device).to(device)
        self.generator=torch.Generator().manual_seed(seed);self.total_steps=0
    def update(self,clients,tensor_hash):
        start=time.time();before=tensor_hash(self.model.state_dict().values());gap=adaptive_gap(clients,self.num_classes,self.device)
        margin=min(gap.max().item(),self.threshold)
        uploads=[(p.detach(),label) for c in clients for label,p in c.protos.items()]
        optimizer=torch.optim.SGD(self.model.parameters(),lr=self.lr);self.model.train();steps=0;epoch_losses=[]
        for _ in range(self.epochs):
            total=0.;count=0
            for proto,y in DataLoader(uploads,self.batch_size,shuffle=True,drop_last=False,generator=self.generator):
                proto=proto.to(self.device);y=y.to(self.device)
                logits=distance_logits(proto,self.model(list(range(self.num_classes))),y,margin)
                loss=F.cross_entropy(logits,y);assert torch.isfinite(loss)
                optimizer.zero_grad();loss.backward();optimizer.step();steps+=1
                total+=loss.item()*len(y);count+=len(y)
            epoch_losses.append(total/count)
        self.total_steps+=steps;self.model.eval()
        with torch.no_grad():bank=self.model(list(range(self.num_classes))).detach()
        assert torch.isfinite(bank).all()
        return {i:bank[i] for i in range(self.num_classes)},dict(initial_hash=before,final_hash=tensor_hash(self.model.state_dict().values()),global_bank_hash=tensor_hash([bank]),uploaded_prototypes=len(uploads),uploaded_labels=[label for _,label in uploads],gap=gap.tolist(),margin=margin,epochs=self.epochs,optimizer_steps=steps,total_optimizer_steps=self.total_steps,epoch_losses=epoch_losses,parameter_count=sum(p.numel() for p in self.model.parameters()),seconds=time.time()-start,prototype_collection='round-start checkpoint eval means',anchors_used=False,test_data_used=False)
