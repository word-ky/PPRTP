"""Preregistered cycle100 endpoint; fixed checkpoints are diagnostics, never selected."""
import json,math,sys
from pathlib import Path
root=Path(sys.argv[1]);base=root/'artifacts/experiment/fedtgp_seed0';read=lambda p:json.loads(Path(p).read_text())
m=read(base/'metadata.json');s=read(base/'split.json');old=Path('research_log/H15A/full/artifacts/experiment/fedtgp_seed0');om=read(old/'metadata.json');a=read(old/'final.json')
assert s==read(old/'split.json') and m['client_initial_hashes']==om['client_initial_hashes']
for key in ('seed','ownership_seed','batch_size','lr','local_epochs','lamda','server_epochs','margin_threshold','server_lr','server_seed','server_initial_hash','server_parameter_count','fedtgp_upstream_sha','mixed_backbone'):assert m[key]==om[key]
assert m['rounds']==100 and m['fedtgp_prototype_timing']=='post_update' and m['prototype_collection']=='post-update eval means' and not m['anchors_used']
rr=[json.loads(x) for x in (base/'rounds.jsonl').read_text().splitlines()];assert len(rr)==100 and rr[-1]==read(base/'final.json')
first=json.loads((old/'rounds.jsonl').read_text().splitlines()[0])
assert rr[0]['client_model_hashes']==first['client_model_hashes'] and rr[0]['batch_hashes']==first['batch_hashes']
assert rr[0]['prototype_bank_hash']!=first['prototype_bank_hash']
previous=m['server_initial_hash'];converged=[];checkpoint_rows=[];receipts={}
for cycle,r in enumerate(rr,1):
 assert r['round']==cycle and r['local_optimizer_steps']==[156]*10 and r['uploaded_class_counts']==s['class_counts']
 t=r['fedtgp_server'];assert t['initial_hash']==previous;previous=t['final_hash']
 assert t['optimizer_steps']==700 and t['total_optimizer_steps']==700*cycle and t['epochs']==100
 assert t['uploaded_prototypes']==200 and sorted(t['uploaded_labels'])==sorted(list(range(100))*2)
 assert t['global_labels']==list(range(100)) and t['distance_logits_finite'] and not t['anchors_used'] and not t['test_data_used'] and not t['pprtp_transport_called']
 assert t['prototype_collection']=='post-update eval means' and len(t['epoch_losses'])==100 and all(math.isfinite(x) for x in t['epoch_losses'])
 assert math.isclose(t['margin'],min(max(t['gap']),100))
 if t['epoch_losses'][-1]<.001:converged.append(dict(cycle=cycle,server_loss=t['epoch_losses'][-1],missing_percent=100*r['metrics']['l2']['missing']))
 if cycle in (10,25,50,100):
  assert read(base/f'checkpoint_cycle{cycle}.json')==r and r['checkpoint_file']==f'checkpoint_cycle{cycle}.pt'
  receipts[str(cycle)]=dict(metrics={k:r['metrics'][k] for k in ('l2','head')},per_client_coverage={k:[c[k]['predicted_class_count'] for c in r['per_client']] for k in ('l2','head')},aggregate_coverage={k:r['predicted_class_counts'][k] for k in ('l2','head')},last_epoch_server_loss=t['epoch_losses'][-1],first_epoch_server_loss=t['epoch_losses'][0],client_steps=1560*cycle,server_steps=700*cycle,semantic_vector_uplink_bytes=409600*cycle,semantic_label_uplink_bytes=1600*cycle,global_proto_downlink_bytes_all_clients=2048000*cycle,wall_seconds=r['elapsed_seconds'],server_seconds=sum(x['fedtgp_server']['seconds'] for x in rr[:cycle]),checkpoint_file=r['checkpoint_file'])
  for name,key in [('official nearest','l2'),('local head','head')]:
   metric=r['metrics'][key];checkpoint_rows.append(f'| {cycle} | {name} | '+' | '.join(f'{100*metric[k]:.6f}' for k in ('seen','missing','all','macro'))+f" | {r['predicted_class_counts'][key]} |")
paired=read('research_log/H12A/full/artifacts/experiment/fedgh_seed0/final.json')['full_data_readout']['pprtp_h07']['metrics'];f=rr[-1]['metrics']['l2']
gaps={k:100*(paired[k]-f[k]) for k in ('seen','missing','all')}
verdict='COMPETITIVE / NOVELTY WARNING' if gaps['missing']<=1 or gaps['all']<=1 else ('STRONG STRESS-TEST EDGE' if gaps['missing']>=3 and gaps['all']>=1 else 'MIXED')
timing={k:100*(rr[9]['metrics']['l2'][k]-a['metrics']['l2'][k]) for k in ('seen','missing','all')}
lines=['# H15-B FedTGP post-update / 100-cycle stress test','','Primary endpoint cycle100; checkpoints10/25/50/100 fixed before run, no best-checkpoint selection.','','| Cycle | Readout | Seen % | Missing % | All % | Macro % | Aggregate classes |','|---|---|---:|---:|---:|---:|---:|']+checkpoint_rows
lines+=['','| Frozen comparator | Seen % | Missing % | All % |','|---|---:|---:|---:|']
for name,metric in [('PPRTP seed0 (10 cycles)',paired),('H15-A round-start FedTGP (10 cycles)',a['metrics']['l2'])]:lines.append('| '+name+' | '+' | '.join(f'{100*metric[k]:.6f}' for k in ('seen','missing','all'))+' |')
lines+=['',f'Frozen endpoint verdict: {verdict}.','PPRTP minus cycle100 FedTGP (pp): '+json.dumps(gaps),'Cycle10 post-update minus H15-A round-start (pp): '+json.dumps(timing),'','Fixed checkpoint receipts (loss is mean over the final server inner epoch):']
for cycle,receipt in receipts.items():lines.append(f'- Cycle{cycle}: '+json.dumps(receipt))
lines+=['','Cycles whose final server-epoch loss is <0.001: '+json.dumps(converged),'Fresh trajectory with historical seed0client/server initialhashes, exactsplit/anchors-excluded/firstbatch and round1clientmodelhashes. Only prototypecollection timing changes at cycle10; cycle100 additionally has10xlocal/server/communication budget. All100 globalprototype distances finite. No anchors/testlabels/PPRTP optimization path.','PPRTP extraanchor uplink5242880bytes remains; no equal-information or communication-efficiency claim. FedTGP server576512parameters not transmitted. Final serverhash: '+previous,'This is one seed and bounded100cycles; do not claim globally converged FedTGP unless its criterion is met, and do not extend/tune to improve results. H15-A artifacts unchanged. Fixed binarycheckpoints stayremote; JSONreceipts and per-round/per-client metrics are committed.']
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');(root/'verification.json').write_text(json.dumps(dict(verdict=verdict,gaps=gaps,timing_control_delta_pp=timing,checkpoints=receipts,server_converged_cycles=converged,split_exact=True,initial_client_server_exact=True,round1_client_models_batches_exact=True,post_update_prototype_effect=True,client_steps=156000,server_steps=70000,server_final_hash=previous,source_sha=m['source_sha']),indent=2),encoding='utf-8')
print('\n'.join(lines[:23]));print(verdict);print('Converged cycles:',converged)
