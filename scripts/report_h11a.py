import json,sys,math
from pathlib import Path
root=Path(sys.argv[1]);base=root/'artifacts/experiment'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
modes=('local','fedproto','fedgh');runs={m:read(base/f'{m}_seed0/final.json') for m in modes}
splits={m:read(base/f'{m}_seed0/split.json') for m in modes};s=splits['local'];assert all(v==s for v in splits.values())
flat=s['anchor_indices']+sum(s['train_indices'],[]);assert len(flat)==len(set(flat))==50000 and sorted(flat)==list(range(50000))
assert s['train_count']==49744 and s['anchor_count']==256 and s['test_indices']==list(range(10000))
history=read('research_log/H02A/full/artifacts/experiment/fedgh_seed0/split.json');assert s['class_sets']==history['class_sets']
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pprtp.full_data import reserve_anchors,index_hash
assert s['anchor_indices']==reserve_anchors(50000) and index_hash(s['anchor_indices'])==s['anchor_indices_sha256']
for ii,hh in zip(s['train_indices'],s['train_index_hashes']):assert index_hash(ii)==hh
for c,owners in s['owners'].items():
 assert owners==[i for i,cs in enumerate(s['class_sets']) if int(c) in cs]
 counts=[s['class_counts'][i][c] for i in owners];assert max(counts)-min(counts)<=1
first=[];runtime={};meta={}
for mode in modes:
 folder=base/f'{mode}_seed0';rr=[json.loads(x) for x in (folder/'rounds.jsonl').read_text().splitlines()];assert len(rr)==10
 first.append(rr[0]);meta[mode]=read(folder/'metadata.json')
 for r in rr:
  assert r['local_optimizer_steps']==[(len(ii)+31)//32 for ii in s['train_indices']]
  for counts,expected in zip(r['uploaded_class_counts'],s['class_counts']):assert counts==expected
  for metric in r['metrics'].values():assert all(math.isfinite(v) for v in metric.values())
  for client in r['per_client']:
   for metric in client.values():assert metric['class_count']==[1000]*10
 runtime[mode]=dict(total_seconds=rr[-1]['elapsed_seconds'],optimizer_steps_total=sum(sum(r['local_optimizer_steps']) for r in rr),
  optimizer_steps_per_client_round=rr[0]['local_optimizer_steps'],local_training_seconds_per_client=[sum(r['local_training_seconds'][i] for r in rr) for i in range(10)])
assert all(r['client_model_hashes']==first[0]['client_model_hashes'] and r['prototype_bank_hash']==first[0]['prototype_bank_hash'] for r in first)
assert len({v['initial_state_sha256'] for v in meta.values()})==len({v['split_sha256'] for v in meta.values()})==1
f=runs['fedgh']['full_data_readout'];p=f['pprtp_h07'];n=f['native_global_prototype_cosine_control']
assert f['same_final_state_exact'] and f['same_raw_means_counts_exact'] and p['state_before']==n['state_before']==p['state_after']==n['state_after']
assert p['state_before']['clients']==runs['fedgh']['client_model_hashes']
for arm in (p,n):
 for flag in ('rng_cpu_unchanged','rng_cuda_unchanged','module_modes_unchanged','existing_gradients_unchanged','cosine_logits_finite'):assert arm[flag]
 for proto in arm['local_prototypes']:assert proto['count']==s['class_counts'][proto['client']][str(proto['label'])]
assert [(v['client'],v['label'],v['count'],v['raw_hash']) for v in p['local_prototypes']]==[(v['client'],v['label'],v['count'],v['raw_hash']) for v in n['local_prototypes']]
rows=[]
for mode,key in [('local','head'),('fedproto','l2'),('fedgh','global_head_post_server')]:rows.append((mode+' deployed '+key,runs[mode]['metrics'][key],runs[mode]['predicted_class_counts'][key]))
rows.extend([('pprtp_h07',p['metrics'],p['predicted_class_count']),('native matched control',n['metrics'],n['predicted_class_count'])])
m=p['metrics'];g=runs['fedgh']['metrics']['global_head_post_server'];fp=runs['fedproto']['metrics']['l2'];gain=100*(m['missing']-n['metrics']['missing'])
gates=dict(missing_at_least15=m['missing']>=.15,missing_beats_baselines5pp=m['missing']>=max(g['missing'],fp['missing'])+.05,all_beats_baselines2pp=m['all']>=max(g['all'],fp['all'])+.02,correspondence_missing_gain10pp=gain>=10,predicted_classes9=p['predicted_class_count']>=9,integrity=True)
verdict='STRONG' if all(gates.values()) else ('WEAK' if m['missing']<.10 or gain<5 else 'POSITIVE BUT NOT STRONG' if gain>=10 else 'INTERMEDIATE: outside explicit strong/positive/weak branches; lead decision required')
lines=['# H11-A full-data seed0 frozen H07 scale validation','','| Arm | Seen % | Missing % | All % | Macro % | Classes |','|---|---:|---:|---:|---:|---:|']
for name,metric,classes in rows:lines.append('| '+name+' | '+' | '.join(f'{100*metric[k]:.6f}' for k in ('seen','missing','all','macro'))+f' | {classes} |')
lines+=['',f'Frozen verdict: {verdict}. Correspondence missing gain: {gain:.6f}pp.', 'Gates: '+json.dumps(gates), '', 'Coverage:49744 disjoint client-training images +256 label-blind anchors =50000 official train. Test:10000 official test. Exact same split/initial model/all-arm round1 hashes.', 'Owners: '+json.dumps(s['owners']), 'Client class counts: '+json.dumps(s['class_counts']), 'Anchor SHA256: '+s['anchor_indices_sha256'], '', 'Runtime and optimizer steps: '+json.dumps(runtime),'Diagnostic elapsed seconds: '+str(f['diagnostic_seconds']), 'Communication: '+json.dumps(f['communication']), 'Forward examples: '+json.dumps(f['forward_examples']), '', 'Global prototype norms: '+json.dumps(f['global_prototype_norms']), 'Global prototype pairwise cosine: '+json.dumps(f['global_prototype_cosine']), 'Canonical Procrustes residual/orthogonality: '+json.dumps(p['alignment']), '', 'Existing common readouts (diagnostics, no tuned probe):']
for mode in modes:lines.append(mode+': '+json.dumps(runs[mode]['metrics']))
lines+=['', 'All-client/per-class counts and predicted-class counts are in final.json. PPRTP/native per-client predicted-class counts derive from their stored prediction histograms. Communication reports naive full float32 affine-map delivery including the identity reference map; shared reference means/identity could avoid redundancy but no compression is claimed. Anchor image distribution is not included in feature payload. Generic CLI train_per_class/test_per_class metadata defaults are ignored with full_data=True; split.json lists actual full coverage. Runtime includes evaluation; local-only training seconds are separately recorded.']
per_client_classes={mode:{key:[v[key]['predicted_class_count'] for v in runs[mode]['per_client']] for key in runs[mode]['per_client'][0]} for mode in modes}
per_client_classes.update(pprtp_h07=[sum(v>0 for v in h['overall']) for h in p['prediction_histograms']['per_client']],native_control=[sum(v>0 for v in h['overall']) for h in n['prediction_histograms']['per_client']])
lines.append('Per-client predicted-class counts: '+json.dumps(per_client_classes))
verification=dict(per_client_predicted_classes=per_client_classes,gates=gates,verdict=verdict,correspondence_missing_gain_pp=gain,runtime=runtime,split_identical=True,round1_identical=True,coverage_exact=True,final_state_and_means_exact=True,source_sha=meta['fedgh']['source_sha'])
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');(root/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
print('\n'.join(lines[:13]));print(json.dumps(verification,indent=2))
