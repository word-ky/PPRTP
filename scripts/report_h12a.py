import hashlib,json,math,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]);base=root/'artifacts/experiment';seed=int(sys.argv[2]) if len(sys.argv)>2 else 0
destination=root if seed==0 else root/f'seed{seed}';destination.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(v):return hashlib.sha256(json.dumps(v,separators=(',',':')).encode()).hexdigest()
modes=('local','fedproto','fedgh');runs={m:read(base/f'{m}_seed{seed}/final.json') for m in modes};splits={m:read(base/f'{m}_seed{seed}/split.json') for m in modes};s=splits['local'];assert all(v==s for v in splits.values())
order=np.random.default_rng(120100).permutation(100).tolist();sets=[[] for _ in range(10)]
for j,c in enumerate(order):sets[j%10].append(c);sets[(j+1)%10].append(c)
sets=[sorted(cs) for cs in sets];assert s['class_sets']==sets and s['class_sets_sha256']==sha(sets) and s['ownership_order']==order and s['ownership_order_sha256']==sha(order)
assert s['ownership_seed']==120100 and s['allocation_rng_seed']==110001
anchors=np.random.default_rng(161803).permutation(50000)[:256].tolist();assert s['anchor_indices']==anchors and s['anchor_indices_sha256']==sha(anchors)
assert s['anchor_selection_label_blind'] and not s['anchor_labels_used']
flat=anchors+sum(s['train_indices'],[]);assert sorted(flat)==list(range(50000)) and len(flat)==len(set(flat))==50000
assert s['train_count']==49744 and s['test_indices']==list(range(10000))
for ii,hh in zip(s['train_indices'],s['train_index_hashes']):assert sha(ii)==hh
for c in range(100):
 owners=[i for i,cs in enumerate(sets) if c in cs];assert len(owners)==2 and s['owners'][str(c)]==owners
 counts=[s['class_counts'][i][str(c)] for i in owners];assert abs(counts[0]-counts[1])<=1
first=[];meta={};runtime={};coverage={}
for mode in modes:
 folder=base/f'{mode}_seed{seed}';rr=[json.loads(x) for x in (folder/'rounds.jsonl').read_text().splitlines()];assert len(rr)==10 and rr[-1]==runs[mode];first.append(rr[0]);meta[mode]=read(folder/'metadata.json')
 assert meta[mode]['num_classes']==100 and meta[mode]['dataset']=='CIFAR100'
 for r in rr:
  assert r['local_optimizer_steps']==[(len(ii)+31)//32 for ii in s['train_indices']]
  assert r['uploaded_class_counts']==s['class_counts']
  for metric in r['metrics'].values():assert all(math.isfinite(v) for v in metric.values())
  for client in r['per_client']:
   for metric in client.values():assert metric['class_count']==[100]*100 and len(metric['prediction_histogram'])==100
 runtime[mode]=dict(elapsed_seconds=rr[-1]['elapsed_seconds'],optimizer_steps_total=sum(sum(r['local_optimizer_steps']) for r in rr),steps_per_client_round=rr[0]['local_optimizer_steps'])
assert len({m['initial_state_sha256'] for m in meta.values()})==len({m['split_sha256'] for m in meta.values()})==1
assert all(r['client_model_hashes']==first[0]['client_model_hashes'] and r['prototype_bank_hash']==first[0]['prototype_bank_hash'] for r in first)
if seed:
 old=Path('research_log/H12A/full/artifacts/experiment')
 assert s==read(old/'local_seed0/split.json')
 assert meta['local']['initial_state_sha256']!=read(old/'local_seed0/metadata.json')['initial_state_sha256']
 if seed==2:assert meta['local']['initial_state_sha256']!=read(base/'local_seed1/metadata.json')['initial_state_sha256']
f=runs['fedgh']['full_data_readout'];p=f['pprtp_h07'];n=f['native_global_prototype_cosine_control'];probe=runs['fedgh']['full_pair_probe'];b=probe['pair_broken_h07']
assert f['same_final_state_exact'] and f['same_raw_means_counts_exact'] and probe['same_raw_means_counts_exact'] and probe['anchor_feature_hashes_exact']
assert p['state_before']['clients']==runs['fedgh']['client_model_hashes']
sig=lambda a:[(v['client'],v['label'],v['count'],v['raw_hash']) for v in a['local_prototypes']]
assert sig(p)==sig(n)==sig(b) and b['anchor_feature_hashes']==probe['paired_anchor_feature_hashes']
assert [r['fixed_points'] for r in b['permutation_receipts']]==[1,0,1,2,1,0,2,3,1]
assert b['alignment'][0]==p['alignment'][0] and b['reference_client_unchanged']
for r in b['permutation_receipts']:
 i=r['client'];perm=np.random.default_rng(314159+i).permutation(256).tolist()
 assert r['permutation']==perm and r['seed']==314159+i and r['permutation_sha256']==sha(perm) and r['multiset_bitwise_unchanged']
 assert sorted(perm)==list(range(256)) and r['original_feature_hash']==b['anchor_feature_hashes'][i]
rows=[]
for mode,key in [('local','head'),('fedproto','l2'),('fedgh','global_head_post_server')]:
 rows.append((mode,runs[mode]['metrics'][key],runs[mode]['predicted_class_counts'][key]));coverage[mode]=[v[key]['predicted_class_count'] for v in runs[mode]['per_client']]
for name,a in [('paired_h07',p),('pair_broken_h07',b),('native_control',n)]:
 assert a['state_before']==a['state_after']==p['state_before'] and a['global_labels']==list(range(100))
 for key in ('rng_cpu_unchanged','rng_cuda_unchanged','module_modes_unchanged','existing_gradients_unchanged','cosine_logits_finite'):assert a[key]
 assert not a['anchor_labels_used'] and not a['test_used_for_transform']
 for proto in a['local_prototypes']:assert proto['count']==s['class_counts'][proto['client']][str(proto['label'])]
 rows.append((name,a['metrics'],a['predicted_class_count']));coverage[name]=[sum(v>0 for v in h['overall']) for h in a['prediction_histograms']['per_client']]
m=p['metrics'];missing_gap=100*(m['missing']-b['metrics']['missing']);native_gap=100*(m['missing']-n['metrics']['missing']);best=max(runs['fedproto']['metrics']['l2']['all'],runs['fedgh']['metrics']['global_head_post_server']['all']);all_gap=100*(m['all']-best)
gates=dict(missing_at_least5=m['missing']>=.05,native_gap_at_least4=native_gap>=4,broken_gap_at_least3=missing_gap>=3,all_gain_at_least1=all_gap>=1)
verdict='FAILURE: external validity unsupported' if m['missing']<.03 or missing_gap<1 else ('STRONG' if all(gates.values()) else 'POSITIVE BUT NOT STRONG')
lines=[f'# H12-A CIFAR100 seed{seed} portability stress test','','| Arm | Seen % | Missing % | All % | Macro % | Aggregate classes |','|---|---:|---:|---:|---:|---:|']
for name,metric,classes in rows:lines.append('| '+name+' | '+' | '.join(f'{100*metric[k]:.6f}' for k in ('seen','missing','all','macro'))+f' | {classes} |')
lines+=['',f'Frozen verdict: {verdict}.',f'Missing paired-minus-broken: {missing_gap:.6f} pp; paired-minus-native: {native_gap:.6f} pp; all gain vs best FedProto/FedGH: {all_gap:.6f} pp.', 'Gates: '+json.dumps(gates),'','Split/initialization/all-arm round1 hashes identical. Exact50000-index coverage:49744 disjoint clienttrain+256 label-blind anchors;10000 evaluation-only test.20classes/client,exactly2owners/class.','Ownership classsets SHA256: '+s['class_sets_sha256'],'Ownership order SHA256: '+s['ownership_order_sha256'],'Anchor SHA256: '+s['anchor_indices_sha256'],'Split-file SHA256: '+meta['fedgh']['split_sha256'],'Classsets: '+json.dumps(sets),'','Per-client predicted-class coverage: '+json.dumps(coverage),'Runtime/steps: '+json.dumps(runtime),'Readout diagnostic seconds: '+str(f['diagnostic_seconds']),'Communication: '+json.dumps(f['communication']),'Forward examples paired/native: '+json.dumps(f['forward_examples']),'Broken control additionally refreshes2560anchor features+49744localfeatures and10000test images/client; same per-readout payload as paired. No optimized deployment/communication-efficiency claim.','', '| Client | Paired centered residual | Broken centered residual |','|---|---:|---:|']
for i,(a,c) in enumerate(zip(p['alignment'],b['alignment'])):lines.append(f"| {i} | {a['centered_residual_after']:.6f} | {c['centered_residual_after']:.6f} |")
lines+=['','All rawmeans/counts,model/server/prototype state and CPU/CUDA RNG/modes/existinggradients are identical acrossreadouts. Exactlegacy fixedpoints[1,0,1,2,1,0,2,3,1], unchangedanchor multisets; permutations/SHA/featurehashes and full residuals/classwise counts in final.json. Client0 reference unchanged. No anchor/testlabels enter transport fitting.','Common baseline readouts (diagnostic only):']
for mode in modes:lines.append(mode+': '+json.dumps(runs[mode]['metrics']))
lines+=['','One seed only; no tuning, seed/graph/anchor sweep or readout selection. Aggregatecoverage is not perclientcoverage. Metadata train_per_class/test_per_class are unused legacy defaults under full_data; actual split receipts are authoritative.']
(destination/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(destination/'verification.json').write_text(json.dumps(dict(verdict=verdict,gates=gates,missing_gap_pp=missing_gap,native_gap_pp=native_gap,all_gap_pp=all_gap,coverage=coverage,runtime=runtime,split_identical=True,initial_state_identical=True,round1_identical=True,coverage_exact=True,final_state_and_raw_means_exact=True,permutation_receipts=b['permutation_receipts'],source_sha=meta['fedgh']['source_sha']),indent=2),encoding='utf-8')
print('\n'.join(lines[:16]))
