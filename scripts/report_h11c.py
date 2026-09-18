import hashlib,json,sys
from pathlib import Path
import numpy as np
root=Path(sys.argv[1]);checks={};gaps=[];lines=['# H11-C full-data pair-breaking control','','| Seed | Readout | Seen % | Missing % | All % | Macro % | Aggregate classes |','|---|---|---:|---:|---:|---:|---:|'];details=[]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
for seed in (0,1,2):
 folder=root/'artifacts/experiment'/f'fedgh_seed{seed}'
 old=Path('research_log/H11A/full' if seed==0 else 'research_log/H11B/full')/'artifacts/experiment'/f'fedgh_seed{seed}'
 split=read(folder/'split.json');assert split==read(old/'split.json')
 metadata=read(folder/'metadata.json');om=read(old/'metadata.json')
 for key in ('initial_state_sha256','split_sha256','upstream_sha','model','optimizer','rounds','batch_size','lr','local_epochs'):assert metadata[key]==om[key],key
 rounds=[json.loads(s) for s in (folder/'rounds.jsonl').read_text().splitlines()];history=[json.loads(s) for s in (old/'rounds.jsonl').read_text().splitlines()]
 assert len(rounds)==len(history)==10
 for a,b in zip(rounds,history):
  for key in ('metrics','per_client','client_model_hashes','prototype_bank_hash','server_head','uploaded_class_counts','local_optimizer_steps'):assert a[key]==b[key],key
 final=read(folder/'final.json');assert final==rounds[-1]
 paired=dict(final['full_data_readout']);paired.pop('diagnostic_seconds');ref=dict(history[-1]['full_data_readout']);ref.pop('diagnostic_seconds');assert paired==ref
 p=paired['pprtp_h07'];n=paired['native_global_prototype_cosine_control'];probe=final['full_pair_probe'];b=probe['pair_broken_h07']
 for key in ('h11_entire_reference_exact','all_online_rounds_exact','split_exact','anchor_feature_hashes_exact','same_raw_means_counts_exact'):assert probe[key]
 assert b['anchor_feature_hashes']==probe['paired_anchor_feature_hashes']
 signature=lambda r:[(q['client'],q['label'],q['count'],q['raw_hash']) for q in r['local_prototypes']]
 assert signature(p)==signature(b)==signature(n)
 assert b['alignment'][0]==p['alignment'][0] and b['reference_client_unchanged']
 receipts=b['permutation_receipts'];assert [r['client'] for r in receipts]==list(range(1,10));assert [r['fixed_points'] for r in receipts]==[1,0,1,2,1,0,2,3,1]
 for r in receipts:
  i=r['client'];perm=np.random.default_rng(314159+i).permutation(256).tolist()
  assert r['seed']==314159+i and r['permutation']==perm and sorted(perm)==list(range(256)) and perm!=list(range(256))
  assert r['permutation_sha256']==hashlib.sha256(json.dumps(perm,separators=(',',':')).encode()).hexdigest()
  assert r['multiset_bitwise_unchanged'] and r['original_feature_hash']==b['anchor_feature_hashes'][i]
 coverage={}
 for name,r in [('paired_h07',p),('pair_broken_h07',b),('native_control',n)]:
  assert r['state_before']==r['state_after']==p['state_before']
  for key in ('rng_cpu_unchanged','rng_cuda_unchanged','module_modes_unchanged','existing_gradients_unchanged'):assert r[key]
  assert not r['anchor_labels_used'] and not r['test_used_for_transform']
  coverage[name]=[sum(v>0 for v in h['overall']) for h in r['prediction_histograms']['per_client']]
  lines.append(f'| {seed} | {name} | '+' | '.join(f"{100*r['metrics'][k]:.6f}" for k in ('seen','missing','all','macro'))+f" | {r['predicted_class_count']} |")
 gap=100*(p['metrics']['missing']-b['metrics']['missing']);agap=100*(p['metrics']['all']-b['metrics']['all']);gaps.append(gap)
 checks[str(seed)]=dict(source_sha=metadata['source_sha'],historical_full_readout_exact=True,all10online_rounds_exact=True,split_exact=True,initial_state_exact=True,same_raw_means_counts=True,state_rng_modes_gradients_unchanged=True,permutation_receipts=receipts,coverage=coverage,missing_gap_pp=gap,all_gap_pp=agap,causal_gate_pass=gap>=8,paired_alignment=p['alignment'],broken_alignment=b['alignment'],elapsed_seconds=final['elapsed_seconds'],diagnostic_seconds=final['full_data_readout']['diagnostic_seconds'])
 details += ['',f'## Seed {seed}',f'Missing gap {gap:.6f} pp; all gap {agap:.6f} pp; gate >=8 pp: {gap>=8}.',f'Per-client class coverage: {json.dumps(coverage)}.','','| Client | Paired centered residual | Broken centered residual |','|---|---:|---:|']
 for i,(a,c) in enumerate(zip(p['alignment'],b['alignment'])):details.append(f"| {i} | {a['centered_residual_after']:.6f} | {c['centered_residual_after']:.6f} |")
passed=sum(g>=8 for g in gaps)
verdict='STOP AND REASSESS: correspondence not adequately attributed' if passed<=1 or min(gaps)<3 else ('3/3 PASS: correct correspondence remains causally important at full-data scale' if passed==3 else '2/3 PASS: seed-sensitive causal effect')
lines+=['',f'Frozen verdict: {verdict}.',f'Missing gaps (pp): {gaps}.', '', 'All three paired/native readouts reproduce historical H11-A/B exactly excluding timing; all ten online rounds, split, initial state and final hashes match. Raw local means/counts and model/server/prototype state match across readouts; RNG, existing gradients and modes unchanged. Anchors and test labels are excluded from transform fitting.', '', 'Legacy permutation seeds314160..314168 unchanged; exact fixed-point counts [1,0,1,2,1,0,2,3,1]. Client8 keeps3/256 (1.171875%) rows, per pre-performance lead amendment f9c70a2. All permutations are bijective and preserve anchor-feature multisets bitwise. Full permutations, SHA receipts and feature hashes are in final.json and verification.json. No tuning or alternate permutations.', '', 'Only FedGH was rerun because archived checkpoints held client0/server head, not all10clients. These are final-state diagnostic readouts, not online-training or communication-efficiency evidence. Aggregate class coverage does not imply every client predicts all10classes.']+details
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(root/'verification.json').write_text(json.dumps(dict(verdict=verdict,passed_seed_count=passed,missing_gaps_pp=gaps,per_seed=checks),indent=2),encoding='utf-8')
print('\n'.join(lines[:21]))
