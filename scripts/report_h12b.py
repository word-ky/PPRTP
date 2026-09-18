import json,sys,statistics
from pathlib import Path
root=Path(sys.argv[1]);checks={};values={};metas=[];splits=[];gaps=[];allgains=[]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
lines=['# H12-B CIFAR100 fixed-split stochastic replication','','| Seed | Readout | Seen % | Missing % | All % | Macro % | Aggregate classes |','|---|---|---:|---:|---:|---:|---:|']
for seed in (0,1,2):
 base=Path('research_log/H12A/full') if seed==0 else root
 checks[str(seed)]=read(base/'verification.json' if seed==0 else base/f'seed{seed}/verification.json')
 folder=base/'artifacts/experiment';runs={m:read(folder/f'{m}_seed{seed}/final.json') for m in ('local','fedproto','fedgh')}
 split=read(folder/f'fedgh_seed{seed}/split.json');splits.append(split);metas.append(read(folder/f'fedgh_seed{seed}/metadata.json'))
 f=runs['fedgh']['full_data_readout'];p=f['pprtp_h07'];n=f['native_global_prototype_cosine_control'];b=runs['fedgh']['full_pair_probe']['pair_broken_h07']
 arms=[(m,runs[m]['metrics'][key],runs[m]['predicted_class_counts'][key]) for m,key in [('local','head'),('fedproto','l2'),('fedgh','global_head_post_server')]]
 arms += [(name,a['metrics'],a['predicted_class_count']) for name,a in [('paired_h07',p),('pair_broken_h07',b),('native_control',n)]]
 for name,m,classes in arms:
  values.setdefault(name,[]).append(m)
  lines.append(f'| {seed} | {name} | '+' | '.join(f'{100*m[k]:.6f}' for k in ('seen','missing','all','macro'))+f' | {classes} |')
 assert checks[str(seed)]['split_identical'] and checks[str(seed)]['initial_state_identical'] and checks[str(seed)]['round1_identical'] and checks[str(seed)]['final_state_and_raw_means_exact']
 gaps.append(checks[str(seed)]['missing_gap_pp']);allgains.append(checks[str(seed)]['all_gap_pp'])
assert splits[0]==splits[1]==splits[2]
assert len({m['initial_state_sha256'] for m in metas})==3 and len({m['split_sha256'] for m in metas})==1
summary={};lines+=['','Mean +/- sample SD (n=3,ddof=1), percentage points:','','| Readout | Seen | Missing | All |','|---|---:|---:|---:|']
for name,mm in values.items():
 summary[name]={k:dict(mean=100*statistics.mean(m[k] for m in mm),std=100*statistics.stdev(m[k] for m in mm)) for k in ('seen','missing','all')}
 lines.append('| '+name+' | '+' | '.join(f"{summary[name][k]['mean']:.6f} +/- {summary[name][k]['std']:.6f}" for k in ('seen','missing','all'))+' |')
gapstats={name:dict(mean=statistics.mean(v),std=statistics.stdev(v)) for name,v in [('paired_minus_broken_missing_pp',gaps),('paired_all_gain_vs_best_FL_pp',allgains)]}
strong=[all(checks[str(seed)]['gates'].values()) for seed in (0,1,2)];count=sum(strong)
failure=count<=1 or any(values['paired_h07'][i]['missing']<.03 or gaps[i]<1 for i in (1,2))
verdict='PAUSE EXPANSION: stochastic robustness unsupported' if failure else ('3/3 STRONG: accept CIFAR100 fixed-split stochastic replication' if count==3 else '2/3 STRONG: CIFAR100 portability is seed-sensitive')
lines+=['',f'Frozen verdict: {verdict}.',f'Seedwise strong: {strong}.',f'Seedwise paired-minus-broken missing gaps (pp): {gaps}.',f'Seedwise paired all gains vs better preregistered FedProto/FedGH (pp): {allgains}.','Gap means +/- sampleSD: '+json.dumps(gapstats),'','Exact same H12-A split, ownershipgraph120100, anchors161803 and allocation110001 acrossallseeds. This is initialization/training-shuffle stochastic replication, NOT ownership-graph replication. Allthree initial model hashes are distinct; within each seed allthree arms share initialization and round1 pairing. No tuning/seed selection.','Initial model SHA256 seed0/1/2: '+json.dumps([m['initial_state_sha256'] for m in metas]),'Exact common split-file SHA256: '+metas[0]['split_sha256'],'','Seed0 is unchanged committed H12-A; seeds1/2 rawoutputs and reports include full residuals, classwise/perclient counts, coverage, steps, hashes, permutation receipts, communication and warnings. Aggregatecoverage does not imply everyclient predicts everyclass. Seen/missing tradeoff remains. No newnumericalablation, architecture, method, online-training or communication-efficiency claim.']
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(root/'verification.json').write_text(json.dumps(dict(verdict=verdict,strong_seed_count=count,strong=strong,summary=summary,gap_summary=gapstats,missing_gaps_pp=gaps,all_gains_pp=allgains,split_exact_across_seeds=True,initial_hashes_distinct=True,initial_hashes=[m['initial_state_sha256'] for m in metas],per_seed=checks),indent=2),encoding='utf-8')
print('\n'.join(lines))
