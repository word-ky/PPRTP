import json,sys
from pathlib import Path
root=Path(sys.argv[1]); folder=root/'artifacts/experiment/fedgh_seed0'
rr=[json.loads(s) for s in (folder/'rounds.jsonl').read_text().splitlines()]
old=[json.loads(s) for s in Path('research_log/H02A/full/artifacts/experiment/fedgh_seed0/rounds.jsonl').read_text().splitlines()]
assert len(rr)==10
for r,h in zip(rr,old):
    assert all(r[k]==h[k] for k in ('metrics','client_model_hashes','prototype_bank_hash','server_head'))
assert all('persistence_probe' not in r for r in rr[:9])
provenance=json.loads((folder/'paired_anchors.json').read_text())
assert provenance==json.loads(Path('research_log/H03A/gate/artifacts/experiment/fedgh_seed0/paired_anchors.json').read_text())
arms=rr[-1]['persistence_probe']
assert arms['no_align_500']['state_before']==arms['paired_500']['state_before']
checks=dict(all_ten_online_rounds_exact=True,exact_anchor_support_reuse=True)
lines=['# H03-C round10 correspondence persistence','', '| Arm | Seen % | Missing % | All % | Macro % | Fit CE before/after | Fit accuracy before/after | Iterations/evaluations |', '|---|---:|---:|---:|---:|---|---|---|']
for name,o in arms.items():
    f=o['fit']
    assert o['state_before']==o['state_after'] and o['rng_cpu_unchanged'] and o['rng_cuda_unchanged'] and o['module_modes_unchanged']
    assert f['max_iter']==500
    checks[name]=dict(state_rng_modes_unchanged=True,fit_adequate=f['after']['accuracy']>=.95)
    fields=[f"{100*o['metrics'][k]:.6f}" for k in ('seen','missing','all','macro')]
    fields += [f"{f['before']['ce']:.9g}/{f['after']['ce']:.9g}",f"{f['before']['accuracy']:.6f}/{f['after']['accuracy']:.6f}",f"{f['n_iter']}/{f['func_evals']}"]
    lines.append('| '+name+' | '+' | '.join(fields)+' |')
b=100*arms['no_align_500']['metrics']['missing'];p=100*arms['paired_500']['metrics']['missing'];o=32.725
q=(p-b)/(o-b) if o>b else None;delta=p-b
if not all(checks[n]['fit_adequate'] for n in arms): verdict='optimizer-limited'
elif q is not None and q>=.5 and delta>=10: verdict='persistent correspondence mechanism'
elif abs(b-o)<=5: verdict='late-round correspondence not needed'
elif (q is not None and q<=.2 or delta<5) and b<o-5: verdict='early-only / degraded mechanism'
else: verdict='intermediate'
checks.update(B10=b,P10=p,O10=o,q10=q,delta10=delta,verdict=verdict)
lines+=['',f'B10={b:.6f}, P10={p:.6f}, O10={o}, q10={q}, delta10={delta:.6f}pp. Frozen branch: {verdict}.', '', '| Client | Centered before | After | Reduction | Orthogonality double | Orthogonality applied | Transform hash |', '|---|---:|---:|---:|---:|---:|---|']
for i,d in enumerate(arms['paired_500']['alignment']):
    lines.append(f'| {i} | '+' | '.join(f'{d[k]:.9g}' for k in ('centered_residual_before','centered_residual_after','relative_residual_reduction','orthogonality_error_double','orthogonality_error_applied'))+' | '+d['transform_hash']+' |')
(root/'verification.json').write_text(json.dumps(checks,indent=2))
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
