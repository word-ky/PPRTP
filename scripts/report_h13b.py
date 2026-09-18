import json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]);checks={};values={};initial=[];batches=[];splits=[];groupgaps={};group_rows=[]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
lines=['# H13-B CIFAR100 mixed-backbone stochastic replication','','| Seed | Arm | Seen % | Missing % | All % | Macro % | Aggregate classes |','|---|---|---:|---:|---:|---:|---:|']
for seed in (0,1,2):
 base=Path('research_log/H13A/full') if seed==0 else root
 v=read(base/'verification.json' if seed==0 else base/f'seed{seed}/verification.json');checks[str(seed)]=v
 folder=base/'artifacts/experiment';m=read(folder/f'local_seed{seed}/metadata.json');initial.append(m['client_initial_states'])
 first=json.loads((folder/f'local_seed{seed}/rounds.jsonl').read_text().splitlines()[0]);batches.append(first['batch_hashes']);splits.append(read(folder/f'local_seed{seed}/split.json'))
 runs={a:read(folder/f'{a}_seed{seed}/final.json') for a in ('local','fedproto','fedgh')};f=runs['fedgh']['full_data_readout']
 arms=[(a,runs[a]['metrics'][key],runs[a]['predicted_class_counts'][key]) for a,key in [('local','head'),('fedproto','l2'),('fedgh','global_head_post_server')]]
 arms += [(a,r['metrics'],r['predicted_class_count']) for a,r in [('paired_h07',f['pprtp_h07']),('pair_broken_h07',runs['fedgh']['full_pair_probe']['pair_broken_h07']),('native_control',f['native_global_prototype_cosine_control'])]]
 for name,metric,classes in arms:
  values.setdefault(name,[]).append(metric)
  lines.append(f'| {seed} | {name} | '+' | '.join(f'{100*metric[k]:.6f}' for k in ('seen','missing','all','macro'))+f' | {classes} |')
 for arch in ('FedAvgCNN','ResNet18'):
  g=v['per_backbone_metrics'];gap=100*(g['paired_h07'][arch]['missing']-g['pair_broken_h07'][arch]['missing']);groupgaps.setdefault(arch,[]).append(gap)
  for arm in ('paired_h07','pair_broken_h07','native_control'):
   metric=g[arm][arch];group_rows.append(f'| {seed} | {arch} | {arm} | '+' | '.join(f'{100*metric[k]:.6f}' for k in ('seen','missing','all'))+f' | {gap:.6f} |')
 for key in ('initial_states_paired_by_client','round1_actual_batches_exact','split_exact_h12a','every_class_cross_architecture','final_state_and_raw_means_exact'):assert v[key]
assert splits[0]==splits[1]==splits[2]
for i in range(10):
 assert len({r[i]['initial_model_hash'] for r in initial})==3
 assert len({tuple(bb[i]) for bb in batches})==3
 assert all(r[i]['architecture']==('FedAvgCNN' if i%2==0 else 'ResNet18') and r[i]['feature_dim']==512 and r[i]['head_shapes']=={'weight':[100,512],'bias':[100]} for r in initial)
summary={};lines+=['','Mean +/- sample SD (n=3,ddof=1), percentage points:','','| Arm | Seen | Missing | All |','|---|---:|---:|---:|']
for name,mm in values.items():
 summary[name]={k:dict(mean=100*statistics.mean(m[k] for m in mm),std=100*statistics.stdev(m[k] for m in mm)) for k in ('seen','missing','all')}
 lines.append('| '+name+' | '+' | '.join(f"{summary[name][k]['mean']:.6f} +/- {summary[name][k]['std']:.6f}" for k in ('seen','missing','all'))+' |')
gaps={key:[checks[str(seed)][key] for seed in (0,1,2)] for key in ('missing_gap_pp','native_gap_pp','all_gap_pp')};gapstats={k:dict(mean=statistics.mean(v),std=statistics.stdev(v)) for k,v in gaps.items()}
strong=[all(checks[str(seed)]['gates'].values()) for seed in (0,1,2)];nstrong=sum(strong)
failure=nstrong<=1 or any(values['paired_h07'][i]['missing']<.03 or gaps['missing_gap_pp'][i]<1 for i in (1,2))
verdict='STOP EXPANSION: mixed-family portability not robust' if failure else ('3/3 STRONG: accept fixed-assignment stochastic mixed-backbone portability' if nstrong==3 else '2/3 STRONG: architecture result is seed-sensitive')
both=all(v>0 for vv in groupgaps.values() for v in vv)
lines+=['',f'Frozen verdict: {verdict}.',f'Seedwise strong: {strong}.','Seedwise frozen gate values: '+json.dumps({seed:v['gates'] for seed,v in checks.items()}),'Seedwise gap values (pp): '+json.dumps(gaps),'Gap mean +/- sampleSD (pp): '+json.dumps(gapstats),'','## Backbone-group qualifications','','| Seed | Backbone | Readout | Seen % | Missing % | All % | Group paired-minus-broken missing pp |','|---|---|---|---:|---:|---:|---:|']+group_rows
lines+=['','Groupwise causal gaps seed0/1/2 (pp): '+json.dumps(groupgaps),f'Positive paired-minus-broken missing gap for both families in ALL3seeds: {both}. This is a claim-qualification diagnostic, not an additional or replacement gate.','', 'Exact H13-A split/ownership/anchors preserved. Per-client initialmodelhashes and actualround1batch orders differ acrossall3seeds; withinseed all3arms pair exactmodel/base/head init and actualbatch orders. Architecture assignment remains alternatingCNN/ResNet18,all512D/head100,oneownerfromeachfamily perclass.', 'Initialization details: '+json.dumps([[dict(client=r['client'],architecture=r['architecture'],initial_model_hash=r['initial_model_hash']) for r in rr] for rr in initial]),'', 'Same-family clients have distinct deterministic initialization; this tests architecture PLUS client-specific initialization heterogeneity. Stochastic replication on onefixedsplit/assignment, not ownership-graph or architecture-assignment replication. Paired/broken/native sharefinalstates/rawmeans; no tuning or solverchoice. Seenaccuracy tradeoff, extra correspondence sideinformation and post-hoc nature remain. Rawresidual scale is not normalized; do not use its magnitude alone as a failurediagnosis. Seed0 reused fromH13A,not rerun. Fullperseed/perclient/backbone results,residuals,steps,payloads,hashes andwarnings accompanythisreport.']
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(root/'verification.json').write_text(json.dumps(dict(verdict=verdict,strong=strong,strong_seed_count=nstrong,summary=summary,gaps=gaps,gap_summary=gapstats,group_causal_gaps=groupgaps,both_families_positive_all_seeds=both,split_exact_across_seeds=True,per_client_initial_hashes_distinct=True,per_client_round1_batches_distinct=True,per_seed=checks),indent=2),encoding='utf-8')
print('\n'.join(lines[:44]));print(json.dumps(groupgaps));print(verdict)
