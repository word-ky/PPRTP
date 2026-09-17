import json,sys
from pathlib import Path
root=Path(sys.argv[1]);folder=root/'artifacts/experiment/fedgh_seed0'
rr=[json.loads(s) for s in (folder/'rounds.jsonl').read_text().splitlines()]
old=[json.loads(s) for s in Path('research_log/H02A/full/artifacts/experiment/fedgh_seed0/rounds.jsonl').read_text().splitlines()]
assert len(rr)==10
for r,h in zip(rr,old):
    assert all(r[k]==h[k] for k in ('metrics','client_model_hashes','prototype_bank_hash','server_head'))
assert all('precision_probe' not in r for r in rr[:9])
history=json.loads(Path('research_log/H05C/gate/artifacts/experiment/fedgh_seed0/final.json').read_text())['helmert_probe']
arms=rr[-1]['precision_probe'];assert arms['anchor_receipt']==history['anchor_receipt']
assert json.loads((folder/'paired_anchors.json').read_text())==json.loads(Path('research_log/H05C/gate/artifacts/experiment/fedgh_seed0/paired_anchors.json').read_text())
names=('rel255_paired_helmert_zscore','rel255_broken_helmert_zscore')
lines=['# H05-D float64 solver-only audit','', '| Arm | Seen % | Missing % | All % | Macro % | Fit % | CE | grad_inf | grad_l2 | Weight/bias norm | Iter/eval |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|']
for name in names:
    original=arms['reproduced_h05c'][name]
    assert {k:v for k,v in original.items() if k!='fixed_features'}==history[name]
    a=arms[name+'_fp64'];f=a['fit'];g=f['final_support']
    assert a['fixed_features']==original['fixed_features']
    for k in ('conditioning','structural_null','gram','state_before','state_after'):
        assert a[k]==original[k]
    if 'permutations' in a: assert a['permutations']==original['permutations']
    assert all(a[k] for k in ('rng_cpu_unchanged','rng_cuda_unchanged','module_modes_unchanged','existing_gradients_unchanged'))
    assert a['casting']['roundtrip_bitwise_equal'] and a['casting']['solver_dtype']=='torch.float64'
    fields=[f"{100*a['metrics'][k]:.6f}" for k in ('seen','missing','all','macro')]
    fields += [f"{100*g['accuracy']:.6f}",f"{g['ce']:.9g}",f"{g['grad_inf']:.9g}",f"{g['grad_l2']:.9g}",f"{g['weight_norm']:.9g}/{g['bias_norm']:.9g}",f"{f['n_iter']}/{f['func_evals']}"]
    lines.append('| '+name+'_fp64 | '+' | '.join(fields)+' |')
p,b=[arms[n+'_fp64'] for n in names];r=100*p['metrics']['missing'];s=100*b['metrics']['missing'];q=r/21.9875;delta=r-s;fit=p['fit']['final_support']
if fit['accuracy']>=.95: verdict='strong relation support' if q>=.7 and delta>=8 else 'clear weak relation' if q<=.4 else 'adequately fit, intermediate'
elif fit['grad_inf']<=1e-6: verdict='stationary underfit: representation/linear-separability limitation supported'
else: verdict='still solver-unresolved'
checks=dict(all_ten_online_exact=True,h05c_entire_historical_output_exact=True,precast_features_exact=True,fp64_only_cast_roundtrip_exact=True,
    state_rng_modes_gradients_unchanged=True,R64=r,S64=s,q_rel_64=q,delta_rel_64=delta,verdict=verdict)
lines+=['',f'R64={r:.6f}%, S64={s:.6f}%, q_rel_64={q:.9g}, delta={delta:.6f}pp. Frozen branch: {verdict}.']
for name in names:
    a=arms[name+'_fp64']
    lines+=['',name+':',json.dumps(a['fixed_features'],indent=2),json.dumps(a['casting'],indent=2),
        'Initial fit: '+json.dumps(a['fit']['before'])]
lines+=['','Both original H05-C float32 arms reproduced exactly, including all metrics, fit diagnostics, hashes and statistics, before the corresponding double fit. Final float32 feature matrices were hashed, then cast only; double-to-float32 roundtrips bitwise equal. All historical online/state/provenance checks passed. No optimizer/iteration/representation change.']
(root/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
