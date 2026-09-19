import json,sys,math
from pathlib import Path
root=Path(sys.argv[1]);base=root/'artifacts/experiment/fedtgp_seed0';read=lambda p:json.loads(Path(p).read_text())
m=read(base/'metadata.json');s=read(base/'split.json');oldroot=Path('research_log/H12A/full');old=oldroot/'artifacts/experiment'
assert s==read(old/'local_seed0/split.json')
assert m['seed']==0 and m['ownership_seed']==120100 and not m['mixed_backbone'] and m['rounds']==10
assert (m['lamda'],m['server_epochs'],m['margin_threshold'],m['server_lr'],m['batch_size'],m['lr'],m['local_epochs'])==(10.,100,100,.01,32,.01,1)
assert m['fedtgp_upstream_sha']=='c77cbbb31eb30d13066cd11f7f4a2e732aeaae24'
assert m['client_initial_hashes']==[read(old/'local_seed0/metadata.json')['initial_state_sha256']]*10 and not m['anchors_used']
rr=[json.loads(x) for x in (base/'rounds.jsonl').read_text().splitlines()];assert len(rr)==10 and rr[-1]==read(base/'final.json')
historical=json.loads((old/'local_seed0/rounds.jsonl').read_text().splitlines()[0]);assert rr[0]['client_model_hashes']==historical['client_model_hashes']
# H12 had no actual batch hash logs; H13-A uses the identical dataset/seed/loader.
audit=Path('research_log/H13A/full/artifacts/experiment/local_seed0')
assert s==read(audit/'split.json')
assert rr[0]['batch_hashes']==json.loads((audit/'rounds.jsonl').read_text().splitlines()[0])['batch_hashes']
previous=m['server_initial_hash'];total=0
for r in rr:
 assert r['local_optimizer_steps']==[156]*10 and r['uploaded_class_counts']==s['class_counts']
 t=r['fedtgp_server'];assert t['initial_hash']==previous;previous=t['final_hash']
 assert t['epochs']==100 and t['optimizer_steps']==700 and t['uploaded_prototypes']==200
 assert sorted(t['uploaded_labels'])==sorted(list(range(100))*2)
 assert t['global_labels']==list(range(100)) and t['distance_logits_finite'] and not t['pprtp_transport_called'] and not t['anchors_used'] and not t['test_data_used']
 assert t['prototype_collection']=='round-start checkpoint eval means' and t['parameter_count']==576512
 assert len(t['epoch_losses'])==100 and all(math.isfinite(x) for x in t['epoch_losses'])
 assert math.isclose(t['margin'],min(max(t['gap']),100))
 total+=t['optimizer_steps'];assert t['total_optimizer_steps']==total
assert total==7000
f=rr[-1];paired=read(old/'fedgh_seed0/final.json')['full_data_readout']['pprtp_h07'];tgp=f['metrics']['l2']
gaps={k:100*(paired['metrics'][k]-tgp[k]) for k in ('seen','missing','all')}
verdict='COMPETITIVE FedTGP / NOVELTY WARNING' if gaps['missing']<=1 or gaps['all']<=1 else ('CLEAR PPRTP EDGE' if gaps['missing']>=3 and gaps['all']>=1 else 'MIXED')
rows=[('FedTGP official nearest',tgp),('FedTGP local head',f['metrics']['head']),('frozen PPRTP',paired['metrics'])]
for mode,key in [('local','head'),('fedproto','l2'),('fedgh','global_head_post_server')]:rows.append(('frozen '+mode,read(old/f'{mode}_seed0/final.json')['metrics'][key]))
lines=['# H15-A matched-protocol FedTGP seed0','','| Arm | Seen % | Missing % | All % | Macro % |','|---|---:|---:|---:|---:|']
for name,metric in rows:lines.append('| '+name+' | '+' | '.join(f'{100*metric[k]:.6f}' for k in ('seen','missing','all','macro'))+' |')
coverage={key:[v[key]['predicted_class_count'] for v in f['per_client']] for key in ('l2','head')}
communication=dict(client_vectors_uplink_bytes_per_cycle=200*512*4,client_labels_uplink_bytes_per_cycle=200*8,global_prototype_downlink_bytes_per_client=100*512*4,global_prototype_downlink_bytes_all_clients_per_cycle=10*100*512*4,server_model_parameters=576512,server_model_transmitted=False,pprtp_anchor_uplink_bytes=5242880)
lines+=['',f'Frozen verdict: {verdict}.','PPRTP minus FedTGP (pp): '+json.dumps(gaps),'Per-client coverage: '+json.dumps(coverage),'Aggregate predicted classes: '+json.dumps(f['predicted_class_counts']),'Communication: '+json.dumps(communication),f'Client optimizer steps15600; server SGD steps7000 (100epochs*7batches*10cycles). Total wall seconds{f["elapsed_seconds"]:.3f}; server-update seconds'+str(sum(r['fedtgp_server']['seconds'] for r in rr)), 'Server first/last epoch losses by cycle: '+json.dumps([[r['fedtgp_server']['epoch_losses'][0],r['fedtgp_server']['epoch_losses'][-1]] for r in rr]),'Initial/final server hashes: '+m['server_initial_hash']+' / '+previous,'','ExactH12split/initialization and round1trainedmodelhashes; actualround1batchhashes equalH13A on exactH12split/seed/loader. Allclasses finite distance predictions. No anchor/testdata in client/server optimization; no PPRTPtransport.','Pinned official disk ordering uploads round-start checkpoint features; retained and tested. Matched batch32/drop_lastFalse/shuffled loaders/10cycles/finalpostserver evaluation and deterministicprivate server RNG are deliberate adaptations. Server margin unweighted-classmeans; individualprototype updates, no samplecount weighting.','Short-budget matched baseline only, not best/converged FedTGP. OfficialREADME discusses >1000 communicationiterations; no extension permitted here. PPRTP uses extra unlabeled same-image correspondence and has a seen-class tradeoff; no communication-efficiency claim. Only one seed. Provenance and adaptations in PROVENANCE.md.']
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');(root/'verification.json').write_text(json.dumps(dict(verdict=verdict,gaps=gaps,coverage=coverage,communication=communication,split_exact=True,initial_exact=True,round1_model_exact=True,round1_batches_exact=True,client_steps=15600,server_steps=total,server_initial_hash=m['server_initial_hash'],server_final_hash=previous,source_sha=m['source_sha']),indent=2),encoding='utf-8')
print('\n'.join(lines))
