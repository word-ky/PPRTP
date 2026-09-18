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
from pprtp.anchor_count import analyze_counts,prefix
from pprtp.relation import analyze_relation
from pprtp.cross_seed import prepare_cross_seed,analyze_cross_seed


def tensor_hash(tensors):
    return hashlib.sha256(b''.join(t.detach().cpu().contiguous().numpy().tobytes()
                                  for t in tensors)).hexdigest()


def owner_compatibility(clients,num_classes=10):
    values = {}
    for label in range(num_classes):
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


def metrics(predictions, labels, seen,num_classes=10):
    count = torch.bincount(labels, minlength=num_classes)
    correct = torch.bincount(labels[predictions == labels], minlength=num_classes)
    mask = torch.zeros(num_classes, dtype=torch.bool)
    mask[seen] = True
    return dict(seen=(correct[mask].sum()/count[mask].sum()).item(),
        missing=(correct[~mask].sum()/count[~mask].sum()).item() if (~mask).any() else None,
        all=(correct.sum()/count.sum()).item(), macro=(correct/count).mean().item(),
        class_correct=correct.tolist(), class_count=count.tolist())


@torch.no_grad()
def evaluate(client, loader, protos, include_histograms=False,num_classes=10):
    client.model.eval()
    predictions = {key: [] for key in ("head", "cosine", "l2")}
    labels = []
    for x,y in loader:
        z = client.model.base(x.to(client.device))
        bank, valid = prototype_bank(protos, num_classes, z)
        predictions['head'].append(client.model.head(z).argmax(1).cpu())
        cosine = F.normalize(z,dim=1) @ F.normalize(bank,dim=1).T
        l2 = (z[:,None,:]-bank[None,:,:]).square().mean(2)
        predictions['cosine'].append(cosine.masked_fill(~valid[None], -torch.inf).argmax(1).cpu())
        predictions['l2'].append(l2.masked_fill(~valid[None], torch.inf).argmin(1).cpu())
        labels.append(y)
    labels = torch.cat(labels)
    result={key: metrics(torch.cat(value), labels, client.class_set,num_classes=num_classes) for key,value in predictions.items()}
    if include_histograms:
        for key,value in predictions.items():
            hist=torch.bincount(torch.cat(value),minlength=num_classes)
            result[key].update(prediction_histogram=hist.tolist(),predicted_class_count=int((hist>0).sum()))
    return result


def run(cfg, mode, seed):
    num_classes=cfg.num_classes
    online=mode in ("pprtp_all_lag1","pprtp_seen_lag1")
    fedgh=mode=="fedgh" or online
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if cfg.full_data:
        from pprtp.full_data import prepare_full,prepare_cifar100
        assert cfg.clients==10 and mode in ('local','fedproto','fedgh')
        if cfg.dataset=='CIFAR100':
            assert seed in (0,1,2) and num_classes==100 and cfg.k==20
            datasets,test,split,full_anchors=prepare_cifar100(cfg.data)
        else:
            assert seed in (0,1,2) and num_classes==10 and cfg.k==2
            datasets,test,split,full_anchors=prepare_full(cfg.data,seed)
    else:
        datasets, test, split = prepare(cfg.data, seed, cfg.clients, cfg.k, cfg.train_per_class, cfg.test_per_class)
    if cfg.full_pair_probe:
        assert cfg.full_data and mode=='fedgh' and cfg.rounds==10 and num_classes==10
        h11=Path('research_log/H11A/full' if seed==0 else 'research_log/H11B/full')/'artifacts/experiment'/f'fedgh_seed{seed}'
        assert json.loads(json.dumps(split))==json.loads((h11/'split.json').read_text())
        h11_rounds=[json.loads(line) for line in (h11/'rounds.jsonl').read_text().splitlines()]
    out = Path(cfg.output) / f"{mode}_seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out/'split.json').write_text(json.dumps(split))
    cnn = FedAvgCNN(in_features=3, num_classes=num_classes, dim=1600)
    head = copy.deepcopy(cnn.fc)
    cnn.fc = torch.nn.Identity()
    args = SimpleNamespace(model=BaseHeadSplit(cnn,head).to(cfg.device),
        algorithm=mode, mode=mode, scale=cfg.scale, dataset='Cifar100' if num_classes==100 else 'Cifar10',device=cfg.device,
        save_folder_name='items',num_classes=num_classes,batch_size=cfg.batch_size,
        local_learning_rate=cfg.lr,local_epochs=cfg.local_epochs,few_shot=0,
        learning_rate_decay_gamma=1.,learning_rate_decay=False,
        lamda=(cfg.seen_lamda if mode=='gpc_seen_match' else
               .002 if mode=='gpc_all_match' or online else cfg.lamda))
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
    if cfg.paired_anchor_probe or cfg.pair_breaking_probe or cfg.persistence_probe or cfg.convexity_probe or cfg.anchor_count_probe or cfg.relation_probe or cfg.conditioning_probe or cfg.helmert_probe or cfg.precision_probe or cfg.precondition_probe or cfg.completion_probe or cfg.class_prototype_probe or cfg.direct_prototype_probe or cfg.local_source_probe:
        oracle_indices=json.loads(Path('research_log/H02C/full/artifacts/experiment/fedgh_seed0/oracle_calibration.json').read_text())['indices']
        saved_support=json.loads(Path('research_log/H02E/full/artifacts/experiment/fedgh_seed0/heldout_owner_support.json').read_text())
        saved_anchors=json.loads(Path('research_log/H03A/gate/artifacts/experiment/fedgh_seed0/paired_anchors.json').read_text()) if (cfg.pair_breaking_probe or cfg.persistence_probe or cfg.convexity_probe or cfg.anchor_count_probe or cfg.relation_probe or cfg.conditioning_probe or cfg.helmert_probe or cfg.precision_probe or cfg.precondition_probe or cfg.completion_probe or cfg.class_prototype_probe or cfg.direct_prototype_probe or cfg.local_source_probe) else None
        anchors,paired_support,paired_receipt=prepare_paired(cfg.data,split,oracle_indices,saved_support['indices'],
            anchor_indices=saved_anchors['indices'] if saved_anchors else None)
        if saved_anchors:
            assert paired_receipt['indices_sha256']==saved_anchors['indices_sha256']
        assert paired_receipt['support_indices_sha256']==saved_support['indices_sha256']
        (out/'paired_anchors.json').write_text(json.dumps(paired_receipt,indent=2))
    if cfg.cross_seed_probe or cfg.direct_cross_seed_probe:
        cross_anchors,cross_support,cross_receipt=prepare_cross_seed(cfg.data,split)
        (out/'cross_seed_provenance.json').write_text(json.dumps(cross_receipt,indent=2))
        if cfg.direct_cross_seed_probe:
            assert seed in (1,2)
            assert cross_receipt==json.loads((Path('research_log/H04B/full/artifacts/experiment')/f'fedgh_seed{seed}'/'cross_seed_provenance.json').read_text())
    if cfg.dual_space_probe or cfg.centered_dual_probe or (cfg.radius_router_probe or cfg.group_refine_probe):
        from pprtp.local_source import local_provenance
        if seed==0:
            from pprtp.online import prepare_online
            dual_anchors,dual_provenance=prepare_online(cfg.data,datasets,split,tensor_hash)
            old=json.loads(Path('research_log/H07A/gate/artifacts/experiment/fedgh_seed0/final.json').read_text())['local_source_probe']
            dual_refs={k:old['localtrain_'+k] for k in ('aligned_global_prototype_cosine','native_global_prototype_cosine_control')}
        else:
            aa,_,cross=prepare_cross_seed(cfg.data,split)
            dual_anchors,receipt=prefix(aa,cross['anchor_indices'],256)
            source=local_provenance(datasets,split,cross['oracle_indices'],cross['support_indices'],receipt['indices'],tensor_hash)
            folder=Path('research_log/H07B/full/artifacts/experiment')/f'fedgh_seed{seed}'
            old=json.loads((folder/'final.json').read_text())['direct_cross_seed_probe']
            assert cross==json.loads((folder/'cross_seed_provenance.json').read_text())
            assert receipt==old['anchor_receipt'] and source==old['local_source']
            dual_provenance=dict(anchor_receipt=receipt,local_source=source,cross_seed=cross)
            dual_refs={k:old['arms'][f'seed{seed}_localtrain_'+k] for k in ('aligned_global_prototype_cosine','native_global_prototype_cosine_control')}
        (out/'dual_provenance.json').write_text(json.dumps(dual_provenance,indent=2))
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
    if online:
        from pprtp.online import prepare_online,build_bank,score_bank
        assert seed==0 and cfg.scale==10. and cfg.rounds==10
        online_anchors,online_source=prepare_online(cfg.data,datasets,split,tensor_hash)
        (out/'online_provenance.json').write_text(json.dumps(online_source,indent=2))
        previous_bank=None
    if fedgh:
        server_head=copy.deepcopy(args.model.head)
        server_optimizer=torch.optim.SGD(server_head.parameters(),lr=.01)
    started=time.time()
    with (out/'rounds.jsonl').open('w') as stream:
        for r in range(cfg.rounds):
            broadcast_receipt=None
            if fedgh and r > 0:
                bases_before=[tensor_hash(c.model.base.state_dict().values()) for c in clients]
                broadcast(server_head,clients)
                head_hash=tensor_hash(server_head.state_dict().values())
                broadcast_receipt=dict(server_head_hash=head_hash,
                    client_head_hashes=[tensor_hash(c.model.head.state_dict().values()) for c in clients],
                    base_hashes=bases_before,
                    bases_unchanged=bases_before==[tensor_hash(c.model.base.state_dict().values()) for c in clients])
                assert broadcast_receipt['bases_unchanged']
                assert all(h==head_hash for h in broadcast_receipt['client_head_hashes'])
            local_steps=[];local_seconds=[]
            for i,client in enumerate(clients):
                gen=torch.Generator().manual_seed(seed*100000+r*100+i)
                loader=DataLoader(datasets[i],batch_size=cfg.batch_size,shuffle=True,drop_last=False,generator=gen)
                client.load_train_data=lambda loader=loader: loader
                local_start=time.time()
                client.train()
                local_seconds.append(time.time()-local_start);local_steps.append(client.optimizer_steps)
            compatibility=owner_compatibility(clients,num_classes)
            protos=aggregate(clients)
            # All global classes covered by construction. Missing validity remains
            # supported and unit-tested at the loss seam, but is not hidden here.
            assert set(protos)==set(range(num_classes))
            bank=torch.stack([protos[c] for c in range(num_classes)])
            if not torch.isfinite(bank).all() or (bank.norm(dim=1)<1e-12).any():
                raise RuntimeError('Nonfinite or zero-norm prototype; H01 stop condition')
            per_client=[evaluate(client,testloader,protos,include_histograms=cfg.full_data,num_classes=num_classes) for client in clients]
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
                    upload_counts=sum(len(c.protos)*8 for c in clients),download_vectors=cfg.clients*num_classes*512*4),
                elapsed_seconds=time.time()-started)
            if fedgh:
                if r == 0 and not cfg.full_data:
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
                    values['global_head_post_server']=evaluate(c,testloader,protos,include_histograms=cfg.full_data,num_classes=num_classes)['head']
                    c.model.head=local_head
                summary['local_head_pre_server']=summary.pop('head')
                summary['global_head_post_server']={k:float(np.mean([v['global_head_post_server'][k] for v in per_client]))
                                                  for k in ('seen','missing','all','macro')}
                record.update(server_head=server,broadcast=broadcast_receipt,
                    historical_round_one_paired=True if r==0 and not cfg.full_data else None,
                    communication_bytes=dict(upload_vectors=sum(len(c.protos) for c in clients)*512*4,upload_labels=sum(len(c.protos) for c in clients)*8,
                        downloaded_head_per_client=sum(p.numel()*p.element_size() for p in server_head.parameters()),
                        downloaded_head_all_clients=len(clients)*sum(p.numel()*p.element_size() for p in server_head.parameters())))
            if online:
                record['batch_hashes']=[c.batch_hashes for c in clients]
                if r==0:
                    old=json.loads(Path('research_log/H02A/full/artifacts/experiment/fedgh_seed0/rounds.jsonl').read_text().splitlines()[0])
                    for key in ('client_model_hashes','prototype_bank_hash','metrics','server_head'):
                        assert json.loads(json.dumps(record[key]))==old[key], key
                    record['historical_round_one_exact']=True
                if previous_bank is not None:
                    assert tensor_hash([clients[0].aligned_bank])==previous_bank['global_hash']
                    assert [tensor_hash(c.aligned_transform) for c in clients]==[d['transform_hash'] for d in previous_bank['alignment']]
                    assert all(c.aligned_bank.grad is None and all(v.grad is None for v in c.aligned_transform) for c in clients)
                    record['training_bank']=dict(source_round=r,global_hash=previous_bank['global_hash'],
                        transform_hashes=[d['transform_hash'] for d in previous_bank['alignment']],frozen_through_epoch=True)
                fresh,transforms,receipt=build_bank(clients,online_anchors,datasets,tensor_hash)
                receipt.update(source_round=r+1,used_in_round=r+2 if r<9 else None,
                    anchor_indices_sha256=online_source['anchor_receipt']['indices_sha256'])
                record['aligned_bank']=receipt
                if r+1 in (2,5,10): record['aligned_direct']=score_bank(clients,test,fresh,transforms,metrics)
                other=Path(cfg.output)/'pprtp_all_lag1_seed0'
                if mode=='pprtp_seen_lag1':
                    first=json.loads((other/'rounds.jsonl').read_text().splitlines()[r])
                    assert record['batch_hashes']==first['batch_hashes']
                    other_meta=json.loads((other/'metadata.json').read_text())
                    assert initial_hash==other_meta['initial_state_sha256'] and metadata['split_sha256']==other_meta['split_sha256']
                    if r==0: assert receipt==first['aligned_bank']
                    if r==1:
                        assert record['diagnostic_client0']['preupdate_feature_hash']==first['diagnostic_client0']['preupdate_feature_hash']
                        assert record['diagnostic_client0']['denominator_feature_gradients']==first['diagnostic_client0']['denominator_feature_gradients']
                for c,t in zip(clients,transforms): c.aligned_bank=fresh;c.aligned_transform=t
                previous_bank=receipt
            if mode == 'fedgh' and (cfg.probe_head or cfg.oracle_head or cfg.owner_sample_probe or cfg.heldout_owner_probe or cfg.paired_anchor_probe or cfg.pair_breaking_probe or cfg.persistence_probe or cfg.convexity_probe or cfg.anchor_count_probe or cfg.relation_probe or cfg.conditioning_probe or cfg.helmert_probe or cfg.precision_probe or cfg.precondition_probe or cfg.completion_probe or cfg.class_prototype_probe or cfg.direct_prototype_probe or cfg.local_source_probe or cfg.cross_seed_probe or cfg.direct_cross_seed_probe or cfg.dual_space_probe or cfg.centered_dual_probe or (cfg.radius_router_probe or cfg.group_refine_probe)):
                historical=Path('research_log/H02A/full/artifacts/experiment')/f'fedgh_seed{seed}'/'rounds.jsonl'
                old=json.loads(historical.read_text().splitlines()[r])
                for key in ('client_model_hashes','prototype_bank_hash','metrics','server_head'):
                    assert json.loads(json.dumps(record[key]))==old[key], f'H02-A online mismatch: round {r+1}, {key}'
                record['historical_online_exact']=True
            if mode=='fedgh' and (cfg.dual_space_probe or cfg.centered_dual_probe or (cfg.radius_router_probe or cfg.group_refine_probe)) and r+1==10:
                from pprtp.direct_prototypes import analyze_direct
                from pprtp.dual_space import analyze_dual
                construction={}
                aligned=analyze_direct(clients,server_head,dual_anchors,datasets,test,tensor_hash,metrics,None,
                    expected_alignment=dual_refs['aligned_global_prototype_cosine']['alignment'],construction_output=construction)
                native=analyze_direct(clients,server_head,dual_anchors,datasets,test,tensor_hash,metrics,None,aligned=False)
                assert aligned==dual_refs['aligned_global_prototype_cosine']
                assert native==dual_refs['native_global_prototype_cosine_control']
                dual=analyze_dual(clients,server_head,test,construction,tensor_hash,metrics)
                assert dual['state_before']==dual['state_after']==aligned['state_before']
                assert dual['global_hash']==aligned['global_hash']
                assert dual['transform_hashes']==[a['transform_hash'] for a in aligned['alignment']]
                for i,hashes in enumerate(dual['owner_raw_hashes']):
                    assert hashes==[p['raw_hash'] for p in aligned['local_prototypes'] if p['client']==i]
                record['dual_space_probe']=dict(aligned_global_prototype_cosine=aligned,native_global_prototype_cosine_control=native,
                    dual_space_owner_seen_aligned_missing=dual,h07_entire_references_exact=True)
                if cfg.group_refine_probe:
                    from pprtp.group_refine import evaluate_refine
                    old=json.loads((Path('research_log/H09A/full/artifacts/experiment')/f'fedgh_seed{seed}'/'final.json').read_text())['dual_space_probe']
                    assert record['dual_space_probe']==old
                    refined=evaluate_refine(clients,server_head,test,construction,tensor_hash,metrics)
                    assert refined['shared_per_client']==aligned['per_client']
                    assert refined['isolation']['state_before']==refined['isolation']['state_after']==dual['state_before']
                    for key in ('global_hash','transform_hashes','owner_raw_hashes'):
                        assert refined[key]==dual[key]
                    for c,v,ref in zip(clients,refined['per_client'],aligned['per_client']):
                        assert sum(v['class_correct'][k] for k in range(10) if k not in c.class_set)==sum(ref['class_correct'][k] for k in range(10) if k not in c.class_set)
                    record['group_refine_probe']=dict(aligned_group_gate_native_owner_refine=refined,h09a_entire_reference_exact=True)
                if cfg.radius_router_probe:
                    from pprtp.router import calibrate_radii,evaluate_router
                    old=json.loads((Path('research_log/H09A/full/artifacts/experiment')/f'fedgh_seed{seed}'/'final.json').read_text())['dual_space_probe']
                    assert record['dual_space_probe']==old
                    radii,calibration=calibrate_radii(clients,server_head,datasets,construction,tensor_hash)
                    assert calibration['isolation']['state_before']==calibration['isolation']['state_after']==dual['state_before']
                    router=evaluate_router(clients,server_head,test,construction,radii,tensor_hash,metrics)
                    assert router['isolation']['state_before']==router['isolation']['state_after']==dual['state_before']
                    assert router['radius_hashes']==calibration['radius_hashes']
                    for c,v,component in zip(clients,router['oracle_seen_missing_router']['per_client'],dual['component_diagnostics']['per_client']):
                        assert sum(v['class_correct'][k] for k in c.class_set)==component['native_owner_seen_only_correct']
                        assert sum(v['class_correct'][k] for k in range(10) if k not in c.class_set)==component['aligned_missing_only_correct']
                    record['radius_router_probe']=dict(loo90_native_accept_else_aligned_missing=router,calibration=calibration,h09a_entire_reference_exact=True)
                if cfg.centered_dual_probe:
                    old=json.loads((Path('research_log/H09A/full/artifacts/experiment')/f'fedgh_seed{seed}'/'final.json').read_text())['dual_space_probe']
                    assert record['dual_space_probe']==old
                    centered=analyze_dual(clients,server_head,test,construction,tensor_hash,metrics,centered=True)
                    control=analyze_dual(clients,server_head,test,construction,tensor_hash,metrics,centered=True,global_only=True)
                    assert centered['state_before']==centered['state_after']==control['state_after']==dual['state_before']
                    record['centered_dual_probe']=dict(centered_dual_owner_seen_aligned_missing=centered,
                        centered_aligned_global_only=control,h09a_entire_reference_exact=True)

            if mode == 'fedgh' and cfg.direct_cross_seed_probe and r+1==10:
                from pprtp.direct_prototypes import analyze_direct
                from pprtp.local_source import local_provenance
                subset,receipt=prefix(cross_anchors,cross_receipt['anchor_indices'],256)
                historical=json.loads((Path('research_log/H04B/full/artifacts/experiment')/f'fedgh_seed{seed}'/'final.json').read_text())['cross_seed_probe']['paired_256_2000']
                assert receipt==historical['anchor_receipt']
                source=local_provenance(datasets,split,cross_receipt['oracle_indices'],cross_receipt['support_indices'],receipt['indices'],tensor_hash)
                aligned=analyze_direct(clients,server_head,subset,datasets,test,tensor_hash,metrics,None,
                    aligned=True,expected_alignment=historical['alignment'])
                native=analyze_direct(clients,server_head,subset,datasets,test,tensor_hash,metrics,None,aligned=False)
                assert aligned['state_before']==native['state_before']==historical['state_before']
                assert [(p['client'],p['label'],p['count'],p['raw_hash']) for p in aligned['local_prototypes']]==[(p['client'],p['label'],p['count'],p['raw_hash']) for p in native['local_prototypes']]
                record['direct_cross_seed_probe']=dict(arms={
                    f'seed{seed}_localtrain_aligned_global_prototype_cosine':aligned,
                    f'seed{seed}_localtrain_native_global_prototype_cosine_control':native},
                    anchor_receipt=receipt,local_source=source,historical_alignment_exact=True,seed=seed)
            if mode == 'fedgh' and cfg.local_source_probe and r+1==10:
                from pprtp.direct_prototypes import analyze_direct
                from pprtp.local_source import local_provenance,source_shift
                subset,receipt=prefix(anchors,paired_receipt['indices'],256)
                historical=json.loads(Path('research_log/H06C/gate/artifacts/experiment/fedgh_seed0/final.json').read_text())['direct_prototype_probe']
                assert receipt==historical['anchor_receipt']
                reference_bank={};local_bank={}
                reference=analyze_direct(clients,server_head,subset,paired_support,test,tensor_hash,metrics,
                    historical['aligned_prototype_learned_head_reference'],aligned=True,bank_output=reference_bank)
                assert reference==historical['aligned_global_prototype_cosine'], 'H07-A stop: H06-C reproduction failed'
                source=local_provenance(datasets,split,oracle_indices,paired_receipt['support_indices'],receipt['indices'],tensor_hash)
                aligned=analyze_direct(clients,server_head,subset,datasets,test,tensor_hash,metrics,None,
                    aligned=True,expected_alignment=reference['alignment'],bank_output=local_bank)
                native=analyze_direct(clients,server_head,subset,datasets,test,tensor_hash,metrics,None,aligned=False)
                assert aligned['state_before']==native['state_before']==reference['state_before']
                assert [(p['client'],p['label'],p['count'],p['raw_hash']) for p in aligned['local_prototypes']]==[(p['client'],p['label'],p['count'],p['raw_hash']) for p in native['local_prototypes']]
                record['local_source_probe']=dict(heldout_aligned_global_prototype_cosine_reference=reference,
                    localtrain_aligned_global_prototype_cosine=aligned,localtrain_native_global_prototype_cosine_control=native,
                    anchor_receipt=receipt,local_source=source,source_shift=source_shift(local_bank['bank'],reference_bank['bank']))
            if mode == 'fedgh' and cfg.direct_prototype_probe and r+1==10:
                from pprtp.class_prototypes import analyze_prototypes
                from pprtp.direct_prototypes import analyze_direct
                subset,receipt=prefix(anchors,paired_receipt['indices'],256)
                historical=json.loads(Path('research_log/H06B/gate/artifacts/experiment/fedgh_seed0/final.json').read_text())['class_prototype_probe']
                assert receipt==historical['anchor_receipt']
                reference=analyze_prototypes(clients,server_head,subset,paired_support,test,tensor_hash,metrics,
                    aligned=True,expected_alignment=historical['aligned_client_class_prototypes']['alignment'])
                assert reference==historical['aligned_client_class_prototypes'], 'H06-C stop: H06-B reproduction failed'
                aligned=analyze_direct(clients,server_head,subset,paired_support,test,tensor_hash,metrics,reference,aligned=True)
                native=analyze_direct(clients,server_head,subset,paired_support,test,tensor_hash,metrics,reference,aligned=False)
                assert aligned['state_before']==native['state_before']==reference['state_before']
                record['direct_prototype_probe']=dict(aligned_prototype_learned_head_reference=reference,
                    aligned_global_prototype_cosine=aligned,native_global_prototype_cosine_control=native,anchor_receipt=receipt)
            if mode == 'fedgh' and cfg.class_prototype_probe and r+1==10:
                from pprtp.class_prototypes import analyze_prototypes
                subset,receipt=prefix(anchors,paired_receipt['indices'],256)
                historical=json.loads(Path('research_log/H04A/gate/artifacts/experiment/fedgh_seed0/final.json').read_text())['anchor_count_probe']['arms']['256']
                assert receipt==historical['anchor_receipt']
                reference=analyze_paired(clients,server_head,subset,paired_support,test,tensor_hash,metrics,
                    max_iter=2000,audit=True,rank_diagnostics=True)
                assert reference==historical['result'], 'H06-B stop: H04-A reproduction failed'
                aligned=analyze_prototypes(clients,server_head,subset,paired_support,test,tensor_hash,metrics,
                    aligned=True,expected_alignment=reference['alignment'])
                native=analyze_prototypes(clients,server_head,subset,paired_support,test,tensor_hash,metrics,aligned=False)
                assert aligned['state_before']==native['state_before']==reference['state_before']
                assert [(p['client'],p['label'],p['count'],p['raw_hash']) for p in aligned['prototypes']]==[(p['client'],p['label'],p['count'],p['raw_hash']) for p in native['prototypes']]
                record['class_prototype_probe']=dict(full_aligned_support_reference=reference,
                    aligned_client_class_prototypes=aligned,native_client_class_prototypes_control=native,anchor_receipt=receipt)
            if mode == 'fedgh' and cfg.completion_probe and r+1==10:
                subset,receipt=prefix(anchors,paired_receipt['indices'],256)
                historical=json.loads(Path('research_log/H04A/gate/artifacts/experiment/fedgh_seed0/final.json').read_text())['anchor_count_probe']['arms']['256']
                assert receipt==historical['anchor_receipt']
                canonical=analyze_paired(clients,server_head,subset,paired_support,test,tensor_hash,metrics,
                    max_iter=2000,audit=True,rank_diagnostics=True)
                assert canonical==historical['result'], 'H06-A stop: canonical H04-A mismatch'
                assert all(d['effective_rank']==255 for d in canonical['rank_diagnostics'])
                arms={'n256_completion_canonical':canonical}
                for k in (1,2,3):
                    alternative=analyze_paired(clients,server_head,subset,paired_support,test,tensor_hash,metrics,
                        max_iter=2000,audit=True,rank_diagnostics=True,completion_arm=k)
                    assert alternative['rank_diagnostics']==canonical['rank_diagnostics']
                    assert alternative['state_before']==canonical['state_before']
                    arms['n256_completion_random'+str(k)]=alternative
                record['completion_probe']=dict(arms=arms,anchor_receipt=receipt,canonical_exact=True)
            if mode == 'fedgh' and cfg.precondition_probe and r+1==10:
                subset,receipt=prefix(anchors,paired_receipt['indices'],256)
                historical=json.loads(Path('research_log/H05D/gate/artifacts/experiment/fedgh_seed0/final.json').read_text())['precision_probe']
                assert receipt==historical['anchor_receipt']
                audits={}
                for broken_arm,arm in ((False,'paired'),(True,'broken')):
                    h=historical['rel255_'+arm+'_helmert_zscore_fp64']
                    result=analyze_relation(clients,server_head,subset,paired_support,test,tensor_hash,metrics,
                        broken=broken_arm,conditioned=True,structural_null=True,
                        expected_features=h['fixed_features'],expected_casting=h['casting'])
                    for key in ('conditioning','structural_null','gram','state_before','state_after'):
                        assert result[key]==h[key]
                    if broken_arm: assert result['permutations']==h['permutations']
                    audits['rel255_'+arm+'_svdprecond_fp64']=result
                record['precondition_probe']=dict(**audits,anchor_receipt=receipt)
            if mode == 'fedgh' and cfg.precision_probe and r+1==10:
                subset,receipt=prefix(anchors,paired_receipt['indices'],256)
                historical=json.loads(Path('research_log/H05C/gate/artifacts/experiment/fedgh_seed0/final.json').read_text())['helmert_probe']
                assert receipt==historical['anchor_receipt']
                audits={};reproduced={}
                for broken_arm,name in ((False,'rel255_paired_helmert_zscore'),(True,'rel255_broken_helmert_zscore')):
                    original=analyze_relation(clients,server_head,subset,paired_support,test,tensor_hash,metrics,
                        broken=broken_arm,conditioned=True,structural_null=True,feature_receipt=True)
                    assert {k:v for k,v in original.items() if k!='fixed_features'}==historical[name]
                    reproduced[name]=original
                    audits[name+'_fp64']=analyze_relation(clients,server_head,subset,paired_support,test,tensor_hash,metrics,
                        broken=broken_arm,conditioned=True,structural_null=True,expected_features=original['fixed_features'])
                record['precision_probe']=dict(**audits,reproduced_h05c=reproduced,anchor_receipt=receipt)
            if mode == 'fedgh' and cfg.helmert_probe and r+1==10:
                subset,receipt=prefix(anchors,paired_receipt['indices'],256)
                paired=analyze_relation(clients,server_head,subset,paired_support,test,tensor_hash,metrics,conditioned=True,structural_null=True)
                broken=analyze_relation(clients,server_head,subset,paired_support,test,tensor_hash,metrics,broken=True,conditioned=True,structural_null=True)
                historical=json.loads(Path('research_log/H05B/gate/artifacts/experiment/fedgh_seed0/final.json').read_text())['conditioning_probe']
                assert receipt==historical['anchor_receipt']
                assert broken['permutations']==historical['rel256_broken_zscore']['permutations']
                for arm,name in ((paired,'rel256_paired_zscore'),(broken,'rel256_broken_zscore')):
                    assert arm['structural_null']['raw_support_hash']==historical[name]['conditioning']['raw_support_hash']
                    assert arm['gram']==historical[name]['gram'] and arm['state_before']==historical[name]['state_before']
                assert paired['state_before']==broken['state_before']
                record['helmert_probe']=dict(rel255_paired_helmert_zscore=paired,rel255_broken_helmert_zscore=broken,anchor_receipt=receipt)
            if mode == 'fedgh' and cfg.conditioning_probe and r+1==10:
                subset,receipt=prefix(anchors,paired_receipt['indices'],256)
                paired=analyze_relation(clients,server_head,subset,paired_support,test,tensor_hash,metrics,conditioned=True)
                broken=analyze_relation(clients,server_head,subset,paired_support,test,tensor_hash,metrics,broken=True,conditioned=True)
                historical=json.loads(Path('research_log/H05A/gate/artifacts/experiment/fedgh_seed0/final.json').read_text())['relation_probe']
                assert receipt==historical['anchor_receipt']
                assert broken['permutations']==historical['rel256_broken']['permutations']
                assert paired['gram']==broken['gram']==historical['rel256_paired']['gram']
                assert paired['state_before']==broken['state_before']==historical['rel256_paired']['state_before']
                record['conditioning_probe']=dict(rel256_paired_zscore=paired,rel256_broken_zscore=broken,anchor_receipt=receipt)
            if mode == 'fedgh' and cfg.relation_probe and r+1==10:
                subset,receipt=prefix(anchors,paired_receipt['indices'],256)
                paired=analyze_relation(clients,server_head,subset,paired_support,test,tensor_hash,metrics)
                broken=analyze_relation(clients,server_head,subset,paired_support,test,tensor_hash,metrics,broken=True)
                assert paired['state_before']==broken['state_before'] and paired['gram']==broken['gram']
                record['relation_probe']=dict(rel256_paired=paired,rel256_broken=broken,anchor_receipt=receipt)
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
            if cfg.full_data:
                assert local_steps==[cfg.local_epochs*((len(d)+cfg.batch_size-1)//cfg.batch_size) for d in datasets]
                record['local_optimizer_steps']=local_steps;record['local_training_seconds']=local_seconds
                record['prediction_histograms']={key:[sum(v[key]['prediction_histogram'][c] for v in per_client) for c in range(num_classes)] for key in per_client[0]}
                record['predicted_class_counts']={key:sum(n>0 for n in hh) for key,hh in record['prediction_histograms'].items()}
                if cfg.full_pair_probe:
                    for key in ('metrics','per_client','client_model_hashes','prototype_bank_hash','server_head','uploaded_class_counts','local_optimizer_steps'):
                        assert json.loads(json.dumps(record[key]))==h11_rounds[r][key], f'H11 reproduction: round{r+1} {key}'
                if mode=='fedgh' and r+1==cfg.rounds:
                    from pprtp.full_data import full_readouts
                    diagnostic_start=time.time()
                    capture={} if cfg.full_pair_probe or num_classes==100 else None
                    record['full_data_readout']=full_readouts(clients,server_head,full_anchors,datasets,test,tensor_hash,metrics,construction_output=capture,num_classes=num_classes)
                    if cfg.full_pair_probe:
                        historical=dict(h11_rounds[-1]['full_data_readout']);historical.pop('diagnostic_seconds')
                        assert record['full_data_readout']==historical, 'Entire H11 readout must reproduce before pair breaking'
                    if cfg.full_pair_probe or num_classes==100:
                        from pprtp.direct_prototypes import analyze_direct
                        broken=analyze_direct(clients,server_head,full_anchors,datasets,test,tensor_hash,metrics,None,broken=True,num_classes=num_classes)
                        paired=record['full_data_readout']['pprtp_h07']
                        assert broken['anchor_feature_hashes']==capture['anchor_feature_hashes']
                        assert broken['state_before']==broken['state_after']==paired['state_before']
                        signature=lambda a:[(p['client'],p['label'],p['count'],p['raw_hash']) for p in a['local_prototypes']]
                        assert signature(broken)==signature(paired)
                        assert broken['alignment'][0]==paired['alignment'][0]
                        assert [p['fixed_points'] for p in broken['permutation_receipts']]==[1,0,1,2,1,0,2,3,1]
                        record['full_pair_probe']=dict(pair_broken_h07=broken,h11_entire_reference_exact=True if cfg.full_pair_probe else None,
                            all_online_rounds_exact=True if cfg.full_pair_probe else None,split_exact=True if cfg.full_pair_probe else None,anchor_feature_hashes_exact=True,
                            same_raw_means_counts_exact=True,paired_anchor_feature_hashes=capture['anchor_feature_hashes'])
                    record['full_data_readout']['diagnostic_seconds']=time.time()-diagnostic_start
                record['elapsed_seconds']=time.time()-started
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
                client.set_protos(protos if mode not in ('local','fedgh') and not online else None)
    if fedgh:
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
    parser.add_argument('--relation-probe',action='store_true')
    parser.add_argument('--conditioning-probe',action='store_true')
    parser.add_argument('--dataset',choices=['CIFAR10','CIFAR100'],default='CIFAR10')
    parser.add_argument('--num-classes',type=int,default=10)
    parser.add_argument('--full-data',action='store_true')
    parser.add_argument('--full-pair-probe',action='store_true')
    parser.add_argument('--group-refine-probe',action='store_true')
    parser.add_argument('--radius-router-probe',action='store_true')
    parser.add_argument('--centered-dual-probe',action='store_true')
    parser.add_argument('--dual-space-probe',action='store_true')
    parser.add_argument('--direct-cross-seed-probe',action='store_true')
    parser.add_argument('--local-source-probe',action='store_true')
    parser.add_argument('--direct-prototype-probe',action='store_true')
    parser.add_argument('--class-prototype-probe',action='store_true')
    parser.add_argument('--completion-probe',action='store_true')
    parser.add_argument('--precondition-probe',action='store_true')
    parser.add_argument('--precision-probe',action='store_true')
    parser.add_argument('--helmert-probe',action='store_true')
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
