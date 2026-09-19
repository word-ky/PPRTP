import hashlib,json,math,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]);base=root/'artifacts/experiment';seed=int(sys.argv[2]) if len(sys.argv)>2 else 0
ownership_seed=int(sys.argv[3]) if len(sys.argv)>3 else 120200
destination=root if seed==0 else root/f'seed{seed}';destination.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(v):return hashlib.sha256(json.dumps(v,separators=(',',':')).encode()).hexdigest()
modes=('local','fedproto','fedgh','fedavg');runs={m:read(base/f'{m}_seed{seed}/final.json') for m in modes};splits={m:read(base/f'{m}_seed{seed}/split.json') for m in modes};s=splits['local'];assert all(v==s for v in splits.values())
order=np.random.default_rng(ownership_seed).permutation(200).tolist();sets=[[] for _ in range(10)]
for j,c in enumerate(order):sets[j%10].append(c)
sets=[sorted(cs) for cs in sets];assert s['class_sets']==sets and s['class_sets_sha256']==sha(sets) and s['ownership_order']==order and s['ownership_order_sha256']==sha(order)
assert s['ownership_seed']==ownership_seed and s['allocation_rng_seed']==110001
anchors=np.random.default_rng(161803).permutation(100000)[:256].tolist();assert s['anchor_indices']==anchors and s['anchor_indices_sha256']==sha(anchors)
assert s['anchor_selection_label_blind'] and not s['anchor_labels_used']
flat=anchors+sum(s['train_indices'],[]);assert sorted(flat)==list(range(100000)) and len(flat)==len(set(flat))==100000
assert s['train_count']==99744 and s['test_indices']==list(range(10000))
for ii,hh in zip(s['train_indices'],s['train_index_hashes']):assert sha(ii)==hh
for c in range(200):
 owners=[i for i,cs in enumerate(sets) if c in cs];assert len(owners)==1 and s['owners'][str(c)]==owners
 assert s['class_counts'][owners[0]][str(c)]>0
first=[];meta={};runtime={};coverage={}
for mode in modes:
 folder=base/f'{mode}_seed{seed}';rr=[json.loads(x) for x in (folder/'rounds.jsonl').read_text().splitlines()];assert len(rr)==10 and rr[-1]==runs[mode];first.append(rr[0]);meta[mode]=read(folder/'metadata.json')
 assert meta[mode]['num_classes']==200 and meta[mode]['dataset']=='TinyImageNet'
 for r in rr:
  assert r['local_optimizer_steps']==[(len(ii)+31)//32 for ii in s['train_indices']]
  assert r['uploaded_class_counts']==s['class_counts']
  for metric in r['metrics'].values():assert all(math.isfinite(v) for v in metric.values())
  for client in r['per_client']:
   for metric in client.values():assert metric['class_count']==[50]*200 and len(metric['prediction_histogram'])==200
 runtime[mode]=dict(elapsed_seconds=rr[-1]['elapsed_seconds'],optimizer_steps_total=sum(sum(r['local_optimizer_steps']) for r in rr),steps_per_client_round=rr[0]['local_optimizer_steps'])
assert len({m['initial_state_sha256'] for m in meta.values()})==len({m['split_sha256'] for m in meta.values()})==1
assert all(r['client_model_hashes']==first[0]['client_model_hashes'] and r['prototype_bank_hash']==first[0]['prototype_bank_hash'] for r in first)
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
for mode,key in [('local','head'),('fedproto','l2'),('fedgh','global_head_post_server'),('fedavg','global_model_post_server')]:
 rows.append((mode,runs[mode]['metrics'][key],runs[mode]['predicted_class_counts'][key]));coverage[mode]=[v[key]['predicted_class_count'] for v in runs[mode]['per_client']]
for name,a in [('paired_h07',p),('pair_broken_h07',b),('native_control',n)]:
 assert a['state_before']==a['state_after']==p['state_before'] and a['global_labels']==list(range(200))
 for key in ('rng_cpu_unchanged','rng_cuda_unchanged','module_modes_unchanged','existing_gradients_unchanged','cosine_logits_finite'):assert a[key]
 assert not a['anchor_labels_used'] and not a['test_used_for_transform']
 for proto in a['local_prototypes']:assert proto['count']==s['class_counts'][proto['client']][str(proto['label'])]
 rows.append((name,a['metrics'],a['predicted_class_count']));coverage[name]=[sum(v>0 for v in h['overall']) for h in a['prediction_histograms']['per_client']]
m=p['metrics'];missing_gap=100*(m['missing']-b['metrics']['missing']);native_gap=100*(m['missing']-n['metrics']['missing']);best=max(runs['fedproto']['metrics']['l2']['all'],runs['fedgh']['metrics']['global_head_post_server']['all']);all_gap=100*(m['all']-best)
gates=dict(missing_at_least2=m['missing']>=.02,native_gap_at_least1_5=native_gap>=1.5,broken_gap_at_least1=missing_gap>=1,aggregate_coverage_at_least160=p['predicted_class_count']>=160,mean_client_coverage_at_least120=float(np.mean(coverage['paired_h07']))>=120)
ready=runs['local']['metrics']['head']['seen']>=.1
verdict='UNDERTRAINED/INCONCLUSIVE' if not ready else 'WEAK/FAILED' if m['missing']<.01 or missing_gap<.5 else 'STRONG' if all(gates.values()) else 'MIXED'
lines=[f'# H17-A Tiny-ImageNet one-owner seed{seed} stress test','','| Arm | Seen % | Missing % | All % | Macro % | Aggregate classes |','|---|---:|---:|---:|---:|---:|']
for name,metric,classes in rows:lines.append('| '+name+' | '+' | '.join(f'{100*metric[k]:.6f}' for k in ('seen','missing','all','macro'))+f' | {classes} |')
lines+=['',f'Frozen verdict: {verdict}.',f'Missing paired-minus-broken: {missing_gap:.6f} pp; paired-minus-native: {native_gap:.6f} pp; all gain vs best FedProto/FedGH: {all_gap:.6f} pp.', 'Gates: '+json.dumps(gates),'','Split/initialization/all-arm round1 hashes identical. Exact100000-index coverage:99744 disjoint clienttrain+256 label-blind anchors;10000 official validation-only evaluation.20classes/client,exactly1owner/class; Tiny ownership order120200, one graph only.','Ownership classsets SHA256: '+s['class_sets_sha256'],'Ownership order SHA256: '+s['ownership_order_sha256'],'Anchor SHA256: '+s['anchor_indices_sha256'],'Split-file SHA256: '+meta['fedgh']['split_sha256'],'Classsets: '+json.dumps(sets),'','Per-client predicted-class coverage: '+json.dumps(coverage),'Runtime/steps: '+json.dumps(runtime),'Readout diagnostic seconds: '+str(f['diagnostic_seconds']),'Communication: '+json.dumps(f['communication']),'Forward examples paired/native: '+json.dumps(f['forward_examples']),'Broken control additionally refreshes2560anchor features+99744localfeatures and10000test images/client; same per-readout payload as paired. No optimized deployment/communication-efficiency claim.','', '| Client | Paired centered residual | Broken centered residual |','|---|---:|---:|']
for i,(a,c) in enumerate(zip(p['alignment'],b['alignment'])):lines.append(f"| {i} | {a['centered_residual_after']:.6f} | {c['centered_residual_after']:.6f} |")
lines+=['','All rawmeans/counts,model/server/prototype state and CPU/CUDA RNG/modes/existinggradients are identical acrossreadouts. Exactlegacy fixedpoints[1,0,1,2,1,0,2,3,1], unchangedanchor multisets; permutations/SHA/featurehashes and full residuals/classwise counts in final.json. Client0 reference unchanged. No anchor/testlabels enter transport fitting.','Common baseline readouts (diagnostic only):']
for mode in modes:lines.append(mode+': '+json.dumps(runs[mode]['metrics']))
lines+=['','One seed only; no tuning, seed/graph/anchor sweep or readout selection. Aggregatecoverage is not perclientcoverage. Metadata train_per_class/test_per_class are unused legacy defaults under full_data; actual split receipts are authoritative.']
assert seed==0 and ownership_seed==120200 and [len(cs) for cs in sets]==[20]*10
assert s['raw_train_count']==100000 and s['raw_val_count']==10000
assert s['raw_train_class_counts']==[500]*200 and s['raw_val_class_counts']==[50]*200
assert not s['official_test_used'] and not s['validation_used_for_fitting']
mapping=s['class_to_idx'];assert list(mapping)==sorted(mapping) and list(mapping.values())==list(range(200))
assert sha(mapping)==s['class_mapping_sha256']
for prefix in ('train','val'):
 files=s[prefix+'_files'];hashes=s[prefix+'_file_hashes']
 assert len(files)==len(set(files))==len(hashes)==(100000 if prefix=='train' else 10000)
 assert files==sorted(files) and sha(files)==s[prefix+'_files_sha256'] and sha(hashes)==s[prefix+'_content_manifest_sha256']
 assert all(len(h)==64 for h in hashes)
assert not set(s['train_files'])&set(s['val_files'])
train_labels=np.array([mapping[Path(name).parts[1]] for name in s['train_files']])
assert np.bincount(train_labels,minlength=200).tolist()==[500]*200
for i,ii in enumerate(s['train_indices']):
 assert sorted(set(train_labels[ii].tolist()))==sets[i]
 assert {str(c):int((train_labels[ii]==c).sum()) for c in sets[i]}==s['class_counts'][i]
assert all(mm['seed']==0 and mm['owners_per_class']==1 and mm['ownership_seed']==120200 and mm['k']==20 and mm['rounds']==10 and mm['batch_size']==32 and mm['local_epochs']==1 and mm['lr']==.01 and not mm['mixed_backbone'] for mm in meta.values())
assert all(mm['input_size']==64 and mm['cnn_dim']==10816 and mm['feature_dim']==512 and mm['head_classes']==200 and mm['model_shape_checked'] and not mm['pretrained'] and not mm['augmentation'] for mm in meta.values())
assert all(r['batch_hashes']==first[0]['batch_hashes'] for r in first)
assert [len(h) for h in first[0]['batch_hashes']]==[(len(ii)+31)//32 for ii in s['train_indices']]
communication={}
for mode in modes:
 rr=[json.loads(x) for x in (base/f'{mode}_seed0/rounds.jsonl').read_text().splitlines()]
 if mode in ('fedgh','fedavg'):communication[mode]={k:sum(r['communication_bytes'][k] for r in rr) for k in rr[0]['communication_bytes']}
 elif mode=='fedproto':communication[mode]={k:sum(r['prototype_payload_bytes'][k] for r in rr) for k in rr[0]['prototype_payload_bytes']}
 else:communication[mode]={'training_network_bytes':0,'prototype_metrics_diagnostic_only':True}
 if mode=='fedavg':
  for rr0 in rr:
   assert rr0['fedavg_server']['all_clients'] and not rr0['fedavg_server']['anchors_used']
   assert rr0['fedavg_server']['weights']==[len(ii)/99744 for ii in s['train_indices']]
   hh=[v['global_model_post_server']['prediction_histogram'] for v in rr0['per_client']]
   assert all(v==hh[0] for v in hh) and all(v['knowledge']==0 for v in rr0['losses'])
warning=runs['fedavg']['metrics']['global_model_post_server']['missing']>=m['missing'] and runs['fedavg']['metrics']['global_model_post_server']['all']>=m['all']
lines+=['',f'Local seen readiness >=10%: {ready}; actual {100*runs["local"]["metrics"]["head"]["seen"]:.6f}%. Global chance0.5%. Readiness precedes mechanism classification.',
 'Training client counts: '+json.dumps(list(map(len,s['train_indices']))),
 'Raw structure 200classes x500train/50officialval; exact lexical ImageFolder class mapping. Per-file SHA256 and ordered-file/index hashes in split.json; val_annotations.txt parsed against train mapping, no official test data used.',
 'Cumulative training communication bytes: '+json.dumps(communication),
 'FedAvg missing/all domination positioning warning: '+str(warning),
 'Pinned PFLlib FedAvgCNN dim10816,64x64RGB,512Dbase/200head,random initialization; ToTensor + Normalize(.5),no resize/augmentation/pretraining. Matched batch32/lr.01/localepoch1/10cycles. PPRTP extra unlabeled same-image correspondence remains. No universal superiority or communication-efficiency claim.',
 'Stop after seed0 regardless of verdict; no result-conditioned tuning or extra rounds.']
(destination/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(destination/'verification.json').write_text(json.dumps(dict(verdict=verdict,training_ready=ready,local_seen=runs['local']['metrics']['head']['seen'],gates=gates,missing_gap_pp=missing_gap,native_gap_pp=native_gap,all_gap_pp=all_gap,coverage=coverage,runtime=runtime,communication=communication,fedavg_positioning_warning=warning,split_identical=True,initial_state_identical=True,round1_identical=True,actual_batches_paired=True,coverage_exact=True,raw_counts_exact=True,raw_file_manifests_verified=True,validation_only=True,model_shape_checked=True,final_state_and_raw_means_exact=True,permutation_receipts=b['permutation_receipts'],source_sha=meta['fedgh']['source_sha']),indent=2),encoding='utf-8')
print('\n'.join(lines[:16]));print('Training readiness:',ready,'FedAvg positioning warning:',warning)
