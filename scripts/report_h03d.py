import json,sys
from pathlib import Path
root=Path(sys.argv[1]);folder=root/'artifacts/experiment/fedgh_seed0'
rr=[json.loads(s) for s in (folder/'rounds.jsonl').read_text().splitlines()]
old=[json.loads(s) for s in Path('research_log/H02A/full/artifacts/experiment/fedgh_seed0/rounds.jsonl').read_text().splitlines()]
assert len(rr)==10
for r,h in zip(rr,old):
    assert all(r[k]==h[k] for k in ('metrics','client_model_hashes','prototype_bank_hash','server_head'))
assert all('convexity_probe' not in r for r in rr[:9])
assert json.loads((folder/'paired_anchors.json').read_text())==json.loads(Path('research_log/H03C/gate/artifacts/experiment/fedgh_seed0/paired_anchors.json').read_text())
arms=rr[-1]['convexity_probe']
historical=json.loads(Path('research_log/H03C/gate/artifacts/experiment/fedgh_seed0/final.json').read_text())['persistence_probe']
assert arms['paired_500']==historical['paired_500']
a=arms['paired_2000'];f=a['fit'];g=f['final_support']
assert a['alignment']==arms['paired_500']['alignment']
assert a['state_before']==arms['paired_500']['state_before']==a['state_after']
assert a['rng_cpu_unchanged'] and a['rng_cuda_unchanged'] and a['module_modes_unchanged']
assert f['after']['ce']==g['ce'] and f['after']['accuracy']==g['accuracy']
assert sum(v['correct'] for v in g['per_client'])/2000==round(g['accuracy'],6)
p=100*a['metrics']['missing'];q=p/32.725
if g['accuracy']>=.95 and q>=.5 and p>=10: verdict='formal persistence gate closes positively'
elif g['accuracy']<.95 and g['grad_inf']<=1e-5: verdict='geometry-limited / near-stationary linear objective (predeclared operational branch)'
elif g['accuracy']<.95: verdict='optimization still unresolved'
else: verdict='fit adequate but positive persistence thresholds not met'
checks=dict(all_ten_online_rounds_exact=True,paired_500_exact=True,alignment_exact=True,indices_exact=True,state_rng_modes_unchanged=True,
    P2000=p,q2000=q,delta2000=p,grad_inf=g['grad_inf'],grad_l2=g['grad_l2'],verdict=verdict)
lines=['# H03-D convexity audit','', '| Arm | Seen % | Missing % | All % | Macro % | CE | Fit % | Iter/eval |','|---|---:|---:|---:|---:|---:|---:|---|']
for name,o in [('no_align_500 reference',historical['no_align_500']),('paired_500 reproduced',arms['paired_500']),('paired_2000',a)]:
    fit=o['fit'];fields=[f"{100*o['metrics'][k]:.6f}" for k in ('seen','missing','all','macro')]
    fields += [f"{fit['after']['ce']:.9g}",f"{100*fit['after']['accuracy']:.6f}",f"{fit['n_iter']}/{fit['func_evals']}"]
    lines.append('| '+name+' | '+' | '.join(fields)+' |')
lines+=['',f'grad_inf={g["grad_inf"]:.9g};grad_l2={g["grad_l2"]:.9g};weight_norm={g["weight_norm"]:.9g};bias_norm={g["bias_norm"]:.9g}.',f'q2000={q:.9g};delta2000={p:.6f}pp;O10=32.725%. Branch: {verdict}.','','| Client | Support correct/count | Accuracy % |','|---|---|---:|']
for i,v in enumerate(g['per_client']): lines.append(f'| {i} | {v["correct"]}/{v["count"]} | {100*v["accuracy"]:.6f} |')
lines+=['','Alignment residuals, orthogonality and every transform hash exactly reproduce H03-C (full arrays in raw final.json). All online records, saved indices, state/RNG/modes match.']
(root/'verification.json').write_text(json.dumps(checks,indent=2))
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
