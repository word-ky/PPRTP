import json,sys
from pathlib import Path
root=Path(sys.argv[1]);folder=root/'artifacts/experiment/fedgh_seed0'
rr=[json.loads(s) for s in (folder/'rounds.jsonl').read_text().splitlines()]
old=[json.loads(s) for s in Path('research_log/H02A/full/artifacts/experiment/fedgh_seed0/rounds.jsonl').read_text().splitlines()]
assert len(rr)==10
for r,h in zip(rr,old):
    assert all(r[k]==h[k] for k in ('metrics','client_model_hashes','prototype_bank_hash','server_head'))
assert all('helmert_probe' not in r for r in rr[:9])
history=json.loads(Path('research_log/H05B/gate/artifacts/experiment/fedgh_seed0/final.json').read_text())['conditioning_probe']
arms=rr[-1]['helmert_probe'];assert arms['anchor_receipt']==history['anchor_receipt']
assert json.loads((folder/'paired_anchors.json').read_text())==json.loads(Path('research_log/H05B/gate/artifacts/experiment/fedgh_seed0/paired_anchors.json').read_text())
names=('rel255_paired_helmert_zscore','rel255_broken_helmert_zscore')
lines=['# H05-C structural null audit','', '| Arm | Seen % | Missing % | All % | Macro % | Fit % | CE | grad_inf | grad_l2 | Weight/bias norm | Iter/eval |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|']
for name,oldname in zip(names,('rel256_paired_zscore','rel256_broken_zscore')):
    a=arms[name];f=a['fit'];g=f['final_support'];h=history[oldname]
    assert a['state_before']==a['state_after']==h['state_before'] and a['gram']==h['gram']
    assert a['structural_null']['raw_support_hash']==h['conditioning']['raw_support_hash']
    assert all(a[k] for k in ('rng_cpu_unchanged','rng_cuda_unchanged','module_modes_unchanged','existing_gradients_unchanged'))
    assert a['conditioning']['input_dtype']=='torch.float32' and len(a['conditioning']['std'])==255
    if 'permutations' in a: assert a['permutations']==h['permutations']
    fields=[f"{100*a['metrics'][k]:.6f}" for k in ('seen','missing','all','macro')]
    fields += [f"{100*g['accuracy']:.6f}",f"{g['ce']:.9g}",f"{g['grad_inf']:.9g}",f"{g['grad_l2']:.9g}",f"{g['weight_norm']:.9g}/{g['bias_norm']:.9g}",f"{f['n_iter']}/{f['func_evals']}"]
    lines.append('| '+name+' | '+' | '.join(fields)+' |')
p,b=[arms[n] for n in names];r=100*p['metrics']['missing'];s=100*b['metrics']['missing'];q=r/21.9875;delta=r-s;fit=p['fit']['final_support']
if fit['accuracy']>=.95: verdict='strong simple relation transport' if q>=.7 and delta>=8 else 'adequately fit but insufficient' if q<.4 else 'adequately fit, intermediate'
elif fit['grad_inf']>1e-5: verdict='still optimization-unresolved'
else: verdict='stationary but underfit'
checks=dict(all_ten_online_exact=True,raw_support_provenance_permutations_exact=True,state_rng_modes_gradients_unchanged=True,
    R255=r,S255=s,q_rel_255=q,delta_rel_255=delta,verdict=verdict,
    paired_fit_adequate=fit['accuracy']>=.95,broken_fit_adequate=b['fit']['after']['accuracy']>=.95)
lines+=['',f'R255={r:.6f}%,S255={s:.6f}%,q_rel_255={q:.9g},delta_rel_255={delta:.6f}pp. Frozen branch: {verdict}.','', '| Arm | Data | Max absolute row sum | RMS row sum | Ones-direction energy / total energy |','|---|---|---:|---:|---:|']
for name in names:
    for data in ('support','test'):
        d=arms[name]['structural_null'][data]
        lines.append(f'| {name} | {data} | {d["row_sum_abs_max"]:.9g} | {d["row_sum_rms"]:.9g} | {d["ones_energy_fraction"]:.9g} |')
lines+=['','| Arm | Stage | Epsilon rule | Rank | smax | Smallest above tolerance | Tolerance | Nonzero condition |','|---|---|---|---:|---:|---:|---:|---:|']
for name in names:
    a=arms[name]
    for stage,d in [('raw256',a['structural_null']['raw_spectrum']),('Helmert255',a['conditioning']['before']),('Helmert255+zscore',a['conditioning']['after'])]:
        for rule,v in [('float64',d),('float32',d['input_aware'])]:
            lines.append(f'| {name} | {stage} | {rule} | {v["rank"]} | {v["singular_max"]:.9g} | {v["singular_min_nonzero"]:.9g} | {v["tolerance"]:.9g} | {v["condition_nonzero"]:.9g} |')
lines+=['','Tolerance=max(matrix_shape)*eps(dtype)*smax;SVD in double on actual float32 input. No data-dependent projection/truncation. Fixed Helmert basis built by closed form in double and cast to float32;training remains float32.']
for name in names: lines+=['',f'{name}: basis SHA256={arms[name]["structural_null"]["basis_hash"]};support-only conditioning SHA256={arms[name]["conditioning"]["statistics_hash"]}.']
(root/'verification.json').write_text(json.dumps(checks,indent=2))
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
