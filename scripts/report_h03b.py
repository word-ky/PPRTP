import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); folder=root/'artifacts/experiment/fedgh_seed0'
rr=[json.loads(s) for s in (folder/'rounds.jsonl').read_text().splitlines()]
old=[json.loads(s) for s in Path('research_log/H02A/full/artifacts/experiment/fedgh_seed0/rounds.jsonl').read_text().splitlines()]
assert len(rr)==2
for r,h in zip(rr,old):
    assert all(r[k]==h[k] for k in ('metrics','client_model_hashes','prototype_bank_hash','server_head'))
provenance=json.loads((folder/'paired_anchors.json').read_text())
original=json.loads(Path('research_log/H03A/gate/artifacts/experiment/fedgh_seed0/paired_anchors.json').read_text())
assert provenance==original
arms=rr[1]['pair_breaking_probe']; paired=arms['paired']
historical=json.loads(Path('research_log/H03A/gate/artifacts/experiment/fedgh_seed0/final.json').read_text())
assert paired['alignment']==historical['paired_anchor_procrustes_probe']['alignment']
lines=['# H03-B pair-breaking control','', 'Frozen anchors/support exactly reused. H02-A online records exact; H03-A paired alignment diagnostics exactly reproduced.',
       '', '| Arm | Seen % | Missing % | All % | Macro % | Fit CE before/after | Fit accuracy before/after | LBFGS iterations/evaluations |',
       '|---|---:|---:|---:|---:|---|---|---|']
checks={}
for arm in ('paired','pair_broken'):
    if arm not in arms: continue
    o=arms[arm];f=o['fit']
    assert o['state_before']==o['state_after'] and o['rng_cpu_unchanged'] and o['rng_cuda_unchanged'] and o['module_modes_unchanged']
    checks[arm]=dict(state_rng_modes_unchanged=True,fit_adequate=f['after']['accuracy']>=.95)
    fields=[f"{100*o['metrics'][k]:.6f}" for k in ('seen','missing','all','macro')]
    fields += [f"{f['before']['ce']:.9g}/{f['after']['ce']:.9g}",f"{f['before']['accuracy']:.6f}/{f['after']['accuracy']:.6f}",f"{f['n_iter']}/{f['func_evals']}"]
    lines.append('| '+arm+' | '+' | '.join(fields)+' |')
if 'pair_broken' in arms:
    p=100*paired['metrics']['missing'];s=100*arms['pair_broken']['metrics']['missing']
    checks.update(q_paired=(p-.0875)/(31.45-.0875),q_broken=(s-.0875)/(31.45-.0875),delta_pair=p-s)
    lines+=['',f"q_paired={checks['q_paired']:.9g}; q_broken={checks['q_broken']:.9g}; delta_pair={p-s:.6f}pp. References B=.0875%,O=31.45%."]
    for rec in arms['pair_broken']['permutations']:
        perm=rec['permutation']
        assert sorted(perm)==list(range(1000)) and rec['fixed_points']==sum(j==v for j,v in enumerate(perm))<=10
        assert rec['multiset_bitwise_unchanged']
        assert rec['permutation_sha256']==hashlib.sha256(json.dumps(perm,separators=(',',':')).encode()).hexdigest()
    lines+=['','| Client | Seed | Fixed points | Permutation hash |','|---|---|---:|---|']
    for rec in arms['pair_broken']['permutations']:
        lines.append(f"| {rec['client']} | {rec['seed']} | {rec['fixed_points']} | {rec['permutation_sha256']} |")
lines+=['','| Arm | Client | Centered before | After | Relative reduction | Orthogonality float64 | Orthogonality float32 |',
        '|---|---|---:|---:|---:|---:|---:|']
for arm in ('paired','pair_broken'):
    if arm not in arms: continue
    for i,d in enumerate(arms[arm]['alignment']):
        lines.append(f'| {arm} | {i} | '+' | '.join(f'{d[k]:.9g}' for k in ('centered_residual_before','centered_residual_after','relative_residual_reduction','orthogonality_error_double','orthogonality_error_applied'))+' |')
(root/'verification.json').write_text(json.dumps(checks,indent=2))
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
