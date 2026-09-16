import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from flcore.trainmodel.models import FedAvgCNN, BaseHeadSplit
from pprtp.client import H01Client, aggregate, prototype_bank
from pprtp.data import prepare


def metrics(predictions, labels, seen):
    count = torch.bincount(labels, minlength=10)
    correct = torch.bincount(labels[predictions == labels], minlength=10)
    mask = torch.zeros(10, dtype=torch.bool)
    mask[seen] = True
    return dict(seen=(correct[mask].sum()/count[mask].sum()).item(),
        missing=(correct[~mask].sum()/count[~mask].sum()).item() if (~mask).any() else None,
        all=(correct.sum()/count.sum()).item(), macro=(correct/count).mean().item(),
        class_correct=correct.tolist(), class_count=count.tolist())


@torch.no_grad()
def evaluate(client, loader, protos):
    client.model.eval()
    predictions = {key: [] for key in ("head", "cosine", "l2")}
    labels = []
    for x,y in loader:
        z = client.model.base(x.to(client.device))
        bank, valid = prototype_bank(protos, 10, z)
        predictions['head'].append(client.model.head(z).argmax(1).cpu())
        cosine = F.normalize(z,dim=1) @ F.normalize(bank,dim=1).T
        l2 = (z[:,None,:]-bank[None,:,:]).square().mean(2)
        predictions['cosine'].append(cosine.masked_fill(~valid[None], -torch.inf).argmax(1).cpu())
        predictions['l2'].append(l2.masked_fill(~valid[None], torch.inf).argmin(1).cpu())
        labels.append(y)
    labels = torch.cat(labels)
    return {key: metrics(torch.cat(value), labels, client.class_set) for key,value in predictions.items()}


def run(cfg, mode, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    datasets, test, split = prepare(cfg.data, seed, cfg.clients, cfg.k, cfg.train_per_class, cfg.test_per_class)
    out = Path(cfg.output) / f"{mode}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out/'split.json').write_text(json.dumps(split))
    cnn = FedAvgCNN(in_features=3, num_classes=10, dim=1600)
    head = copy.deepcopy(cnn.fc)
    cnn.fc = torch.nn.Identity()
    args = SimpleNamespace(model=BaseHeadSplit(cnn,head).to(cfg.device),
        algorithm=mode, mode=mode, scale=cfg.scale, dataset='Cifar10',device=cfg.device,
        save_folder_name='items',num_classes=10,batch_size=cfg.batch_size,
        local_learning_rate=cfg.lr,local_epochs=cfg.local_epochs,few_shot=0,
        learning_rate_decay_gamma=1.,learning_rate_decay=False,lamda=cfg.lamda)
    clients=[]
    for i, ds in enumerate(datasets):
        client=H01Client(args,i,len(ds),len(test),train_slow=False,send_slow=False)
        client.class_set=split['class_sets'][i]
        clients.append(client)
    # Upstream constructor resets global seed to 0; all batch shuffles below use
    # explicit client/round generators, independent of constructor/evaluation RNG.
    initial_hash=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in args.model.state_dict().values())).hexdigest()
    metadata=vars(cfg).copy()
    metadata.update(mode=mode,seed=seed,source_sha=os.environ.get('PPRTP_SOURCE_SHA','unknown'),
        upstream_sha='0169ba7e412c9856a08bb3faefab1e35f538a3c1', torch=torch.__version__,
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(cfg.device) if cfg.device.startswith('cuda') else None,
        model='PFLlib FedAvgCNN 512D', optimizer='SGD, no momentum/decay',
        prototype_rule='sample-count-weighted online raw feature means; no EMA',
        initial_state_sha256=initial_hash,split_sha256=hashlib.sha256((out/'split.json').read_bytes()).hexdigest())
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    testloader=DataLoader(test,batch_size=128,shuffle=False)
    started=time.time()
    with (out/'rounds.jsonl').open('w') as stream:
        for r in range(cfg.rounds):
            for i,client in enumerate(clients):
                gen=torch.Generator().manual_seed(seed*100000+r*100+i)
                loader=DataLoader(datasets[i],batch_size=cfg.batch_size,shuffle=True,drop_last=False,generator=gen)
                client.load_train_data=lambda loader=loader: loader
                client.train()
            protos=aggregate(clients)
            # All ten classes covered by construction. Missing validity remains
            # supported and unit-tested at the loss seam, but is not hidden here.
            assert set(protos)==set(range(10))
            bank=torch.stack([protos[c] for c in range(10)])
            if not torch.isfinite(bank).all() or (bank.norm(dim=1)<1e-12).any():
                raise RuntimeError('Nonfinite or zero-norm prototype; H01 stop condition')
            per_client=[evaluate(client,testloader,protos) for client in clients]
            summary={readout:{metric:float(np.mean([v[readout][metric] for v in per_client]))
                              for metric in ('seen','missing','all','macro')}
                      for readout in ('head','cosine','l2')}
            record=dict(round=r+1,metrics=summary,per_client=per_client,
                losses=[c.losses for c in clients],diagnostic_client0=clients[0].diagnostic,
                prototype_cosine=(F.normalize(bank,dim=1)@F.normalize(bank,dim=1).T).cpu().tolist(),
                prototype_norms=bank.norm(dim=1).tolist(),
                uploaded_class_counts=[c.proto_counts for c in clients],
                prototype_payload_bytes=dict(upload_vectors=sum(len(c.protos)*512*4 for c in clients),
                    upload_counts=sum(len(c.protos)*8 for c in clients),download_vectors=cfg.clients*10*512*4),
                elapsed_seconds=time.time()-started)
            stream.write(json.dumps(record)+'\n'); stream.flush()
            print(json.dumps(dict(mode=mode,seed=seed,round=r+1,metrics=summary,elapsed=record['elapsed_seconds'])),flush=True)
            for client in clients:
                client.set_protos(protos if mode!='local' else None)
    torch.save(dict(model=clients[0].model.state_dict(),global_protos={k:v.cpu() for k,v in protos.items()},
                    client=0,class_set=clients[0].class_set),out/'client0.pt')
    (out/'final.json').write_text(json.dumps(record,indent=2))
    return record


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--modes',nargs='+',default=['local','fedproto','gpc'])
    parser.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--clients',type=int,default=10)
    parser.add_argument('--k',type=int,default=2)
    parser.add_argument('--train-per-class',type=int,default=100)
    parser.add_argument('--test-per-class',type=int,default=100)
    parser.add_argument('--rounds',type=int,default=10)
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--local-epochs',type=int,default=1)
    parser.add_argument('--lr',type=float,default=.01)
    parser.add_argument('--lamda',type=float,default=1.)
    parser.add_argument('--scale',type=float,default=10.)
    cfg=parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    for seed in cfg.seeds:
        for mode in cfg.modes:
            run(cfg,mode,seed)


if __name__=='__main__':
    main()
