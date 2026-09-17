import json,sys
from pathlib import Path
root=Path(sys.argv[1]);folder=root/'artifacts/experiment/fedgh_seed0'
rr=[json.loads(s) for s in (folder/'rounds.jsonl').read_text().splitlines()]
old=[json.loads(s) for s in Path('research_log/H02A/full/artifacts/experiment/fedgh_seed0/rounds.jsonl').read_text().splitlines()]
assert len(rr)==10
for r,h in zip(rr,old):
    assert all(r[k]==h[k] for k in ('metrics','client_model_hashes','prototype_bank_hash','server_head'))
assert all('conditioning_probe' not in r for r in rr[:9])
historical=json.loads(Path('research_log/H05A/gate/artifacts/experiment/fedgh_seed0/final.json').read_text())['relation_probe']
assert json.loads((folder/'paired_anchors.json').read_text())==json.loads(Path('research_log/H05A/gate/artifacts/experiment/fedgh_seed0/paired_anchors.json').read_text())
arms=rr[-1]['conditioning_probe'];assert arms['anchor_receipt']==historical['anchor_receipt']
paired=arms['rel256_paired_zscore'];broken=arms['rel256_broken_zscore']
assert broken['permutations']==historical['rel256_broken']['permutations']
lines=['# H05-B invertible conditioning','', '| Arm | Seen % | Missing % | All % | Macro % | Fit % | CE before/after | grad_inf | grad_l2 | Weight/bias norm | Iter/eval |', '|---|---:|---:|---:|---:|---:|---|---:|---:|---|---|']
for name in ('rel256_paired_zscore','rel256_broken_zscore'):
    a=arms[name];f=a['fit'];g=f['final_support'];c=a['conditioning']
    assert a['state_before']==a['state_after']==historical['rel256_paired']['state_before']
    assert a['gram']==historical['rel256_paired']['gram']
    assert all(a[k] for k in ('rng_cpu_unchanged','rng_cuda_unchanged','module_modes_unchanged','existing_gradients_unchanged'))
    assert c['input_dtype']=='torch.float32' and min(c['std'])>0 and not c['labels_used'] and not c['test_used']
    fields=[f"{100*a['metrics'][k]:.6f}" for k in ('seen','missing','all','macro')]
    fields += [f"{100*g['accuracy']:.6f}",f"{f['before']['ce']:.9g}/{g['ce']:.9g}",f"{g['grad_inf']:.9g}",f"{g['grad_l2']:.9g}",f"{g['weight_norm']:.9g}/{g['bias_norm']:.9g}",f"{f['n_iter']}/{f['func_evals']}"]
    lines.append('| '+name+' | '+' | '.join(fields)+' |')
r=100*paired['metrics']['missing'];s=100*broken['metrics']['missing'];q=r/21.9875;delta=r-s
fit=paired['fit']['final_support']
if fit['accuracy']>=.95:
    verdict='strong relation signal' if q>=.7 and delta>=8 else 'simple relation insufficient' if q<.4 else 'intermediate'
elif fit['grad_inf']<=1e-5: verdict='near-stationary; insufficient shared linear support fit'
else: verdict='optimization remains unresolved after fixed conditioning'
checks=dict(all_ten_online_exact=True,provenance_permutations_gram_exact=True,state_rng_modes_gradients_unchanged=True,
    Rz=r,Sz=s,q_rel_z=q,delta_rel_z=delta,paired_fit_adequate=fit['accuracy']>=.95,
    broken_fit_adequate=broken['fit']['after']['accuracy']>=.95,verdict=verdict)
lines+=['',f'Rz={r:.6f}%,Sz={s:.6f}%,q_rel_z={q:.9g},delta_rel_z={delta:.6f}pp. Frozen branch: {verdict}.',
    '', '| Arm | Matrix | Std min/median/max | Singular max/min-nonzero | Rank | Tolerance | Condition nonzero | Abs max | RMS |', '|---|---|---|---|---:|---:|---:|---:|---:|']
hash_lines=[]
for name in ('rel256_paired_zscore','rel256_broken_zscore'):
    c=arms[name]['conditioning']
    for stage in ('before','after'):
        d=c[stage]
        lines.append(f'| {name} | {stage} | {d["std_min"]:.9g}/{d["std_median"]:.9g}/{d["std_max"]:.9g} | {d["singular_max"]:.9g}/{d["singular_min_nonzero"]:.9g} | {d["rank"]} | {d["tolerance"]:.9g} | {d["condition_nonzero"]:.9g} | {d["abs_max"]:.9g} | {d["rms"]:.9g} |')
    hash_lines+=['',f'{name} conditioning statistics SHA256: {c["statistics_hash"]}; raw support SHA256: {c["raw_support_hash"]}.','']
lines+=hash_lines
lines+=['SVD diagnostic tolerance: max(matrix_shape)*eps64*smax. SVD uses the actual uncentered matrix before/after zscore, float64 diagnostics only. Std diagnostics are recomputed in double (median=0.5 quantile); actual mean/std and inputs remain float32. Full mean/std vectors in raw JSON; no labels or test samples used for statistics.']
(root/'verification.json').write_text(json.dumps(checks,indent=2))
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
