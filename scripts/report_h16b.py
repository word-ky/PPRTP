"""H16 fixed one-owner three-seed summary; seed0 is immutable H16-A evidence."""
import json,statistics,sys
from pathlib import Path
root=Path(sys.argv[1]);checks={};values={};initial=[];batches=[];splits=[]
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
lines=['# H16-B one-owner CIFAR100 stochastic replication','','| Seed | Arm | Seen % | Missing % | All % | Macro % | Aggregate coverage |','|---|---|---:|---:|---:|---:|---:|']
for seed in (0,1,2):
 source=Path('research_log/H16A/full') if seed==0 else root
 checks[str(seed)]=read(source/'verification.json' if seed==0 else source/f'seed{seed}/verification.json')
 base=source/'artifacts/experiment'
 runs={mode:read(base/f'{mode}_seed{seed}/final.json') for mode in ('local','fedproto','fedgh','fedavg')}
 meta=read(base/f'local_seed{seed}/metadata.json');initial.append(meta['initial_state_sha256']);splits.append(read(base/f'local_seed{seed}/split.json'))
 first=json.loads((base/f'local_seed{seed}/rounds.jsonl').read_text().splitlines()[0]);batches.append(first['batch_hashes'])
 f=runs['fedgh']['full_data_readout']
 arms=[(mode,runs[mode]['metrics'][key],runs[mode]['predicted_class_counts'][key]) for mode,key in [('local','head'),('fedproto','l2'),('fedgh','global_head_post_server'),('fedavg','global_model_post_server')]]
 arms += [(name,a['metrics'],a['predicted_class_count']) for name,a in [('paired_h07',f['pprtp_h07']),('pair_broken_h07',runs['fedgh']['full_pair_probe']['pair_broken_h07']),('native_control',f['native_global_prototype_cosine_control'])]]
 for name,m,coverage in arms:
  values.setdefault(name,[]).append(m)
  lines.append(f'| {seed} | {name} | '+' | '.join(f'{100*m[k]:.6f}' for k in ('seen','missing','all','macro'))+f' | {coverage} |')
 v=checks[str(seed)]
 for key in ('split_identical','initial_state_identical','round1_identical','coverage_exact','final_state_and_raw_means_exact','actual_batches_paired'):assert v[key]
 assert all(r['optimizer_steps_total']==15590 for r in v['runtime'].values())
assert splits[0]==splits[1]==splits[2] and len(set(initial))==3
for client in range(10):assert len({tuple(b[client]) for b in batches})==3
summary={};lines+=['','Mean +/- sample standard deviation (n=3, ddof=1), percentage points.','','| Arm | Seen | Missing | All |','|---|---:|---:|---:|']
for name,mm in values.items():
 summary[name]={k:dict(mean=100*statistics.mean(m[k] for m in mm),std=100*statistics.stdev(m[k] for m in mm)) for k in ('seen','missing','all')}
 lines.append('| '+name+' | '+' | '.join(f"{summary[name][k]['mean']:.6f} +/- {summary[name][k]['std']:.6f}" for k in ('seen','missing','all'))+' |')
gaps={'paired_minus_broken_missing_pp':[checks[str(i)]['missing_gap_pp'] for i in (0,1,2)],'paired_minus_native_missing_pp':[checks[str(i)]['native_gap_pp'] for i in (0,1,2)]}
for k in ('missing','all'):gaps['paired_minus_fedavg_'+k+'_pp']=[100*(p[k]-a[k]) for p,a in zip(values['paired_h07'],values['fedavg'])]
gapstats={k:dict(mean=statistics.mean(v),std=statistics.stdev(v)) for k,v in gaps.items()}
strong=[all(checks[str(i)]['gates'].values()) for i in (0,1,2)]
verdict='3/3 STRONG' if all(strong) else 'STOP: one-owner replication is not 3/3 STRONG'
warning=[checks[str(i)]['fedavg_positioning_warning'] for i in (0,1,2)]
lines+=['',f'Frozen verdict: {verdict}.','Seedwise verdicts: '+json.dumps({i:v['verdict'] for i,v in checks.items()}),'Seedwise gates: '+json.dumps({i:v['gates'] for i,v in checks.items()}),'Gap values (pp): '+json.dumps(gaps),'Gap mean +/- sample SD: '+json.dumps(gapstats),'FedAvg missing/all domination warnings seed0/1/2: '+json.dumps(warning),'','Exact same H16-A split, owners1/order120100,256 anchors and49744 non-anchor examples. Initial hashes differ across all3seeds; actual first-round minibatch hashes differ for every client across3seeds and pair among4arms within eachseed. Fourarms15590localsteps each; readouts share frozen finalstate/rawmeans/counts and anchorfeature multisets. No labels enter transport fitting.','Initial hashes seed0/1/2: '+json.dumps(initial),'Per-seed coverage, runtime, communication, class counts/correct counts, transformations and integrity receipts remain in individual reports/raw artifacts. Seed0 reused unchanged; no seed selection, tuning, rerun or new dataset.','FedAvg is a homogeneous shared full-model baseline without correspondence information. PPRTP is personalized post-hoc and uses extra same-image anchors; preserve the seen/missing tradeoff and FedAvg comparison, do not claim universal accuracy or communication superiority.','Stop after reporting both seeds regardless of outcome; await lead decision before Tiny-ImageNet.']
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(root/'verification.json').write_text(json.dumps(dict(verdict=verdict,strong=strong,strong_seed_count=sum(strong),summary=summary,gaps=gaps,gap_summary=gapstats,fedavg_positioning_warning=warning,initial_hashes=initial,initial_hashes_distinct=True,actual_batches_distinct_per_client=True,split_exact_h16a=True,per_seed=checks),indent=2),encoding='utf-8')
print('\n'.join(lines))
