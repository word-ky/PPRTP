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
from pprtp.fedgh import broadcast, train_server, fit_probe
from pprtp.oracle import calibration, analyze
from pprtp.owner_probe import provenance, analyze_owner
from pprtp.heldout import prepare_heldout
from pprtp.paired import prepare_paired, analyze_paired
from pprtp.anchor_count import analyze_counts
from pprtp.cross_seed import prepare_cross_seed,analyze_cross_seed


def tensor_hash(tensors):
    return hashlib.sha256(b''.join(t.detach().cpu().contiguous().numpy().tobytes()
                                  for t in tensors)).hexdigest()


def owner_compatibility(clients):
    values = {}
    for label in range(10):
        owners = [c for c in clients if label in c.protos]
        if len(owners) == 2:
            values[label] = dict(owners=[c.id for c in owners], cosine=F.cosine_similarity(
                owners[0].protos[label][None], owners[1].protos[label][None]).item())
    cosines = [v['cosine'] for v in values.values()]
    return dict(per_class=values, mean=float(np.mean(cosines)), min=min(cosines), max=max(cosines)) if cosines else None


def check_round_one(records):
    reference = records[0]
    for record in records[1:]:
        assert record['client_model_hashes'] == reference['client_model_hashes'], 'Round-1 client mismatch'
        assert record['prototype_bank_hash'] == reference['prototype_bank_hash'], 'Round-1 prototype mismatch'


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
        learning_rate_decay_gamma=1.,learning_rate_decay=False,
        lamda=(cfg.seen_lamda if mode=='gpc_seen_match' else
               .002 if mode=='gpc_all_match' else cfg.lamda))
    clients=[]
    for i, ds in enumerate(datasets):
        client=H01Client(args,i,len(ds),len(test),train_slow=False,send_slow=False)
        client.class_set=split['class_sets'][i]
        clients.append(client)
    # Upstream constructor resets global seed to 0; all batch shuffles below use
    # explicit client/round generators, independent of constructor/evaluation RNG.
    initial_hash=tensor_hash(args.model.state_dict().values())
    metadata=vars(cfg).copy()
    metadata.update(mode=mode,seed=seed,lamda=args.lamda,source_sha=os.environ.get('PPRTP_SOURCE_SHA','unknown'),
        upstream_sha='0169ba7e412c9856a08bb3faefab1e35f538a3c1', torch=torch.__version__,
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(cfg.device) if cfg.device.startswith('cuda') else None,
        model='PFLlib FedAvgCNN 512D', optimizer='SGD, no momentum/decay',
        prototype_rule='sample-count-weighted online raw feature means; no EMA',
        initial_state_sha256=initial_hash,split_sha256=hashlib.sha256((out/'split.json').read_bytes()).hexdigest())
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    testloader=DataLoader(test,batch_size=128,shuffle=False)
    if cfg.paired_anchor_probe or cfg.pair_breaking_probe or cfg.persistence_probe or cfg.convexity_probe or cfg.anchor_count_probe:
        oracle_indices=json.loads(Path('research_log/H02C/full/artifacts/experiment/fedgh_seed0/oracle_calibration.json').read_text())['indices']
        saved_support=json.loads(Path('research_log/H02E/full/artifacts/experiment/fedgh_seed0/heldout_owner_support.json').read_text())
        saved_anchors=json.loads(Path('research_log/H03A/gate/artifacts/experiment/fedgh_seed0/paired_anchors.json').read_text()) if (cfg.pair_breaking_probe or cfg.persistence_probe or cfg.convexity_probe or cfg.anchor_count_probe) else None
        anchors,paired_support,paired_receipt=prepare_paired(cfg.data,split,oracle_indices,saved_support['indices'],
            anchor_indices=saved_anchors['indices'] if saved_anchors else None)
        if saved_anchors:
            assert paired_receipt['indices_sha256']==saved_anchors['indices_sha256']
        assert paired_receipt['support_indices_sha256']==saved_support['indices_sha256']
        (out/'paired_anchors.json').write_text(json.dumps(paired_receipt,indent=2))
    if cfg.cross_seed_probe:
        cross_anchors,cross_support,cross_receipt=prepare_cross_seed(cfg.data,split)
        (out/'cross_seed_provenance.json').write_text(json.dumps(cross_receipt,indent=2))
    if cfg.heldout_owner_probe:
        oracle_indices=json.loads(Path('research_log/H02C/full/artifacts/experiment/fedgh_seed0/oracle_calibration.json').read_text())['indices']
        heldout_data,heldout_receipt=prepare_heldout(cfg.data,split,oracle_indices)
        (out/'heldout_owner_support.json').write_text(json.dumps(heldout_receipt,indent=2))
    if cfg.owner_sample_probe:
        oracle_indices=json.loads(Path('research_log/H02C/full/artifacts/experiment/fedgh_seed0/oracle_calibration.json').read_text())['indices']
        owner_receipt=provenance(datasets,split,oracle_indices)
        (out/'owner_samples.json').write_text(json.dumps(owner_receipt,indent=2))
    if cfg.oracle_head:
        oracle_data,oracle_receipt=calibration(cfg.data,split)
        (out/'oracle_calibration.json').write_text(json.dumps(oracle_receipt,indent=2))
    if mode == 'fedgh':
        server_head=copy.deepcopy(args.model.head)
        server_optimizer=torch.optim.SGD(server_head.parameters(),lr=.01)
    started=time.time()
    with (out/'rounds.jsonl').open('w') as stream:
        for r in range(cfg.rounds):
            broadcast_receipt=None
            if mode == 'fedgh' and r > 0:
                bases_before=[tensor_hash(c.model.base.state_dict().values()) for c in clients]
                broadcast(server_head,clients)
                head_hash=tensor_hash(server_head.state_dict().values())
                broadcast_receipt=dict(server_head_hash=head_hash,
                    client_head_hashes=[tensor_hash(c.model.head.state_dict().values()) for c in clients],
                    base_hashes=bases_before,
                    bases_unchanged=bases_before==[tensor_hash(c.model.base.state_dict().values()) for c in clients])
                assert broadcast_receipt['bases_unchanged']
                assert all(h==head_hash for h in broadcast_receipt['client_head_hashes'])
            for i,client in enumerate(clients):
                gen=torch.Generator().manual_seed(seed*100000+r*100+i)
                loader=DataLoader(datasets[i],batch_size=cfg.batch_size,shuffle=True,drop_last=False,generator=gen)
                client.load_train_data=lambda loader=loader: loader
                client.train()
            compatibility=owner_compatibility(clients)
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
                owner_prototype_compatibility=compatibility,
                client_model_hashes=[tensor_hash(c.model.state_dict().values()) for c in clients],
                prototype_bank_hash=tensor_hash([bank]),
                losses=[c.losses for c in clients],diagnostic_client0=clients[0].diagnostic,
                prototype_cosine=(F.normalize(bank,dim=1)@F.normalize(bank,dim=1).T).cpu().tolist(),
                prototype_norms=bank.norm(dim=1).tolist(),
                uploaded_class_counts=[c.proto_counts for c in clients],
                prototype_payload_bytes=dict(upload_vectors=sum(len(c.protos)*512*4 for c in clients),
                    upload_counts=sum(len(c.protos)*8 for c in clients),download_vectors=cfg.clients*10*512*4),
                elapsed_seconds=time.time()-started)
            if mode == 'fedgh':
                if r == 0:
                    historical=Path('research_log/H01B/receipts')/f'fedproto_seed{seed}'/'rounds.jsonl'
                    check_round_one([json.loads(historical.read_text().splitlines()[0]),record])
                before=[tensor_hash(c.model.base.state_dict().values()) for c in clients]
                head_before=tensor_hash(server_head.state_dict().values())
                server=train_server(server_head,server_optimizer,clients)
                server.update(hash_before=head_before,hash_after=tensor_hash(server_head.state_dict().values()),
                    bases_unchanged=before==[tensor_hash(c.model.base.state_dict().values()) for c in clients])
                assert server['bases_unchanged'] and server['hash_before']!=server['hash_after']
                for c,values in zip(clients,per_client):
                    values['local_head_pre_server']=values.pop('head')
                    local_head=c.model.head
                    c.model.head=server_head
                    values['global_head_post_server']=evaluate(c,testloader,protos)['head']
                    c.model.head=local_head
                summary['local_head_pre_server']=summary.pop('head')
                summary['global_head_post_server']={k:float(np.mean([v['global_head_post_server'][k] for v in per_client]))
                                                  for k in ('seen','missing','all','macro')}
                record.update(server_head=server,broadcast=broadcast_receipt,
                    historical_round_one_paired=True if r==0 else None,
                    communication_bytes=dict(upload_vectors=20*512*4,upload_labels=20*8,
                        downloaded_head_per_client=sum(p.numel()*p.element_size() for p in server_head.parameters()),
                        downloaded_head_all_clients=len(clients)*sum(p.numel()*p.element_size() for p in server_head.parameters())))
            if mode == 'fedgh' and (cfg.probe_head or cfg.oracle_head or cfg.owner_sample_probe or cfg.heldout_owner_probe or cfg.paired_anchor_probe or cfg.pair_breaking_probe or cfg.persistence_probe or cfg.convexity_probe or cfg.anchor_count_probe or cfg.cross_seed_probe):
                historical=Path('research_log/H02A/full/artifacts/experiment')/f'fedgh_seed{seed}'/'rounds.jsonl'
                old=json.loads(historical.read_text().splitlines()[r])
                for key in ('client_model_hashes','prototype_bank_hash','metrics','server_head'):
                    assert json.loads(json.dumps(record[key]))==old[key], f'H02-A online mismatch: round {r+1}, {key}'
                record['historical_online_exact']=True
            if mode == 'fedgh' and cfg.cross_seed_probe and r+1==10:
                record['cross_seed_probe']=analyze_cross_seed(clients,server_head,cross_anchors,cross_support,test,
                    tensor_hash,metrics,cross_receipt['anchor_indices'])
            if mode == 'fedgh' and cfg.anchor_count_probe and r+1==10:
                historical=json.loads(Path('research_log/H03D/gate/artifacts/experiment/fedgh_seed0/final.json').read_text())['convexity_probe']['paired_2000']
                record['anchor_count_probe']=analyze_counts(clients,server_head,anchors,paired_support,test,
                    tensor_hash,metrics,paired_receipt['indices'],historical)
            if mode == 'fedgh' and cfg.convexity_probe and r+1==10:
                historical=json.loads(Path('research_log/H03C/gate/artifacts/experiment/fedgh_seed0/final.json').read_text())['persistence_probe']['paired_500']
                paired=analyze_paired(clients,server_head,anchors,paired_support,test,tensor_hash,metrics,
                    max_iter=500,expected_alignment=historical['alignment'])
                assert paired==historical, 'H03-C paired500 reproduction mismatch'
                record['convexity_probe']=dict(paired_500=paired,paired_500_exact=True,
                    paired_2000=analyze_paired(clients,server_head,anchors,paired_support,test,tensor_hash,metrics,
                        max_iter=2000,expected_alignment=historical['alignment'],audit=True))
            if mode == 'fedgh' and cfg.persistence_probe and r+1==10:
                record['persistence_probe']=dict(
                    no_align_500=analyze_owner(clients,server_head,paired_support,test,tensor_hash,metrics,max_iter=500),
                    paired_500=analyze_paired(clients,server_head,anchors,paired_support,test,tensor_hash,metrics,max_iter=500))
                assert record['persistence_probe']['no_align_500']['state_before']==record['persistence_probe']['paired_500']['state_before']
            if mode == 'fedgh' and cfg.pair_breaking_probe and r+1==2:
                historical=json.loads(Path('research_log/H03A/gate/artifacts/experiment/fedgh_seed0/final.json').read_text())
                paired=analyze_paired(clients,server_head,anchors,paired_support,test,tensor_hash,metrics,
                    max_iter=500,expected_alignment=historical['paired_anchor_procrustes_probe']['alignment'])
                record['pair_breaking_probe']=dict(paired=paired,paired_alignment_reproduced=True)
                if paired['fit']['after']['accuracy']<.95:
                    stream.write(json.dumps(record)+'\n'); stream.flush()
                    raise RuntimeError('H03-B stop: paired support fit below95%; optimizer-limited')
                record['pair_breaking_probe']['pair_broken']=analyze_paired(clients,server_head,anchors,paired_support,test,
                    tensor_hash,metrics,broken=True,max_iter=500)
            if mode == 'fedgh' and cfg.paired_anchor_probe and r+1==2:
                record['paired_anchor_procrustes_probe']=analyze_paired(clients,server_head,anchors,paired_support,test,tensor_hash,metrics)
            if mode == 'fedgh' and cfg.heldout_owner_probe and r+1 in (2,10):
                record['heldout_owner_probe']=analyze_owner(clients,server_head,heldout_data,test,tensor_hash,metrics)
            if mode == 'fedgh' and cfg.owner_sample_probe and r+1 in (2,10):
                record['owner_sample_probe']=analyze_owner(clients,server_head,datasets,test,tensor_hash,metrics)
            if mode == 'fedgh' and cfg.oracle_head and r+1 in (1,2,10):
                record['oracle']=analyze(clients,server_head,oracle_data,test,tensor_hash,metrics)
            if mode == 'fedgh' and cfg.probe_head:
                online_before=[tensor_hash(c.model.state_dict().values()) for c in clients]
                server_before=tensor_hash(server_head.state_dict().values())
                probe,info=fit_probe(server_head,clients)
                info['head_hash']=tensor_hash(probe.state_dict().values())
                for c,values in zip(clients,per_client):
                    local_head=c.model.head
                    c.model.head=probe
                    values['probe_head_postfit']=evaluate(c,testloader,protos)['head']
                    c.model.head=local_head
                summary['probe_head_postfit']={k:float(np.mean([v['probe_head_postfit'][k] for v in per_client]))
                                             for k in ('seen','missing','all','macro')}
                info.update(client_hashes_before=online_before,
                    client_hashes_after=[tensor_hash(c.model.state_dict().values()) for c in clients],
                    server_hash_before=server_before,server_hash_after=tensor_hash(server_head.state_dict().values()),
                    historical_online_exact=True)
                assert info['client_hashes_before']==info['client_hashes_after']
                assert info['server_hash_before']==info['server_hash_after']
                record['probe']=info
            stream.write(json.dumps(record)+'\n'); stream.flush()
            if mode == 'fedgh' and cfg.probe_head and record['probe']['after']['accuracy'] < .95:
                raise RuntimeError('H02-B stop: probe prototype accuracy below 95%; preserve negative result')
            if r == 0:
                prior = []
                for other in cfg.modes:
                    path = Path(cfg.output)/f'{other}_seed{seed}'/'rounds.jsonl'
                    if path.exists():
                        prior.append(json.loads(path.read_text().splitlines()[0]))
                check_round_one(prior)
            print(json.dumps(dict(mode=mode,seed=seed,round=r+1,metrics=summary,elapsed=record['elapsed_seconds'])),flush=True)
            for client in clients:
                client.set_protos(protos if mode not in ('local','fedgh') else None)
    if mode == 'fedgh':
        torch.save(server_head.state_dict(),out/'server_head.pt')
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
    parser.add_argument('--seen-lamda',type=float,default=.002)
    parser.add_argument('--scale',type=float,default=10.)
    parser.add_argument('--probe-head',action='store_true')
    parser.add_argument('--oracle-head',action='store_true')
    parser.add_argument('--owner-sample-probe',action='store_true')
    parser.add_argument('--heldout-owner-probe',action='store_true')
    parser.add_argument('--paired-anchor-probe',action='store_true')
    parser.add_argument('--pair-breaking-probe',action='store_true')
    parser.add_argument('--persistence-probe',action='store_true')
    parser.add_argument('--convexity-probe',action='store_true')
    parser.add_argument('--anchor-count-probe',action='store_true')
    parser.add_argument('--cross-seed-probe',action='store_true')
    cfg=parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    for seed in cfg.seeds:
        for mode in cfg.modes:
            run(cfg,mode,seed)
        records=[json.loads((Path(cfg.output)/f'{mode}_seed{seed}'/'rounds.jsonl').read_text().splitlines()[0])
                 for mode in cfg.modes]
        check_round_one(records)
        (Path(cfg.output)/f'round_one_pairing_seed{seed}.json').write_text(json.dumps(
            dict(passed=True,modes=cfg.modes,seed=seed,client_model_hashes=records[0]['client_model_hashes'],
                 prototype_bank_hash=records[0]['prototype_bank_hash']),indent=2))


if __name__=='__main__':
    main()
