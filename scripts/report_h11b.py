import json,sys,statistics
from pathlib import Path
root=Path(sys.argv[1]);rows=[];checks={};values={};anchors=[]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
lines=['# H11-B full-data cross-seed replication','','| Seed | Arm | Seen % | Missing % | All % | Macro % | Classes |','|---|---|---:|---:|---:|---:|---:|']
for seed in (0,1,2):
 base=Path('research_log/H11A/full') if seed==0 else root
 verify=read(base/'verification.json' if seed==0 else base/f'seed{seed}/verification.json');checks[str(seed)]=verify
 folder=base/'artifacts/experiment';runs={m:read(folder/f'{m}_seed{seed}/final.json') for m in ('local','fedproto','fedgh')}
 split=read(folder/f'fedgh_seed{seed}/split.json');anchors.append(split['anchor_indices'])
 history=read(f'research_log/H02A/full/artifacts/experiment/fedgh_seed{seed}/split.json');assert split['class_sets']==history['class_sets']
 if seed:assert history==read(f'research_log/H04B/full/artifacts/experiment/fedgh_seed{seed}/split.json')
 arms=[(m,runs[m]['metrics'][key],runs[m]['predicted_class_counts'][key]) for m,key in [('local','head'),('fedproto','l2'),('fedgh','global_head_post_server')]]
 f=runs['fedgh']['full_data_readout']
 arms += [(name,f[key]['metrics'],f[key]['predicted_class_count']) for name,key in [('pprtp','pprtp_h07'),('native','native_global_prototype_cosine_control')]]
 for name,metric,classes in arms:
  values.setdefault(name,[]).append(metric)
  lines.append(f'| {seed} | {name} | '+' | '.join(f"{100*metric[k]:.6f}" for k in ('seen','missing','all','macro'))+f' | {classes} |')
 assert verify['gates']['integrity']
assert anchors[0]==anchors[1]==anchors[2]
lines+=['','Mean +/- sample standard deviation (n=3, ddof=1), percentage points:','','| Arm | Seen | Missing | All |','|---|---:|---:|---:|']
summary={}
for name,mm in values.items():
 summary[name]={k:dict(mean=100*statistics.mean(m[k] for m in mm),std=100*statistics.stdev(m[k] for m in mm)) for k in ('seen','missing','all')}
 lines.append('| '+name+' | '+' | '.join(f"{summary[name][k]['mean']:.6f} +/- {summary[name][k]['std']:.6f}" for k in ('seen','missing','all'))+' |')
strong=sum(all(v['gates'].values()) for v in checks.values());gains=[v['correspondence_missing_gain_pp'] for v in checks.values()]
verdict='STOP AND REASSESS' if strong<=1 or min(gains)<5 else ('3/3 STRONG: full-data replication accepted; await lead' if strong==3 else '2/3 STRONG: positive but seed-sensitive; no tuning')
lines+=['',f'Frozen verdict: {verdict}.', 'Per-seed strong: '+str([all(v['gates'].values()) for v in checks.values()]), 'Per-seed correspondence missing gain (pp): '+str(gains), '', 'All anchors exactly identical to historical H11-A seed0; ownership inherited from each historical seed. Detailed per-seed gates, provenance, classwise counts, residuals, communication, forward costs, runtime and state isolation are in seed1/seed2 RESULTS.md and raw final.json. Seed0 is unchanged committed H11-A, not rerun. Same deployed readouts; no selection or tuning.']
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(root/'verification.json').write_text(json.dumps(dict(strong_seed_count=strong,verdict=verdict,anchor_indices_equal=True,summary=summary,per_seed=checks),indent=2),encoding='utf-8')
print('\n'.join(lines))
