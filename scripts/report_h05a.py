import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);folder=root/'artifacts/experiment/fedgh_seed0'
rr=[json.loads(s) for s in (folder/'rounds.jsonl').read_text().splitlines()]
old=[json.loads(s) for s in Path('research_log/H02A/full/artifacts/experiment/fedgh_seed0/rounds.jsonl').read_text().splitlines()]
assert len(rr)==10
for r,h in zip(rr,old):
    assert all(r[k]==h[k] for k in ('metrics','client_model_hashes','prototype_bank_hash','server_head'))
assert all('relation_probe' not in r for r in rr[:9])
prov=json.loads((folder/'paired_anchors.json').read_text())
assert prov==json.loads(Path('research_log/H04A/gate/artifacts/experiment/fedgh_seed0/paired_anchors.json').read_text())
arms=rr[-1]['relation_probe'];paired=arms['rel256_paired'];broken=arms['rel256_broken']
historical=json.loads(Path('research_log/H04A/gate/artifacts/experiment/fedgh_seed0/final.json').read_text())['anchor_count_probe']['arms']['256']
assert arms['anchor_receipt']==historical['anchor_receipt']
assert paired['state_before']==broken['state_before'] and paired['gram']==broken['gram']
lines=['# H05-A fixed relation coordinates','', '| Arm | Seen % | Missing % | All % | Macro % | Fit % | CE before/after | grad_inf | grad_l2 | Weight/bias norm | Iter/eval |', '|---|---:|---:|---:|---:|---:|---|---:|---:|---|---|']
checks=dict(all_ten_online_exact=True,provenance_exact=True,state_rng_modes_gradients_unchanged=True)
for name in ('rel256_paired','rel256_broken'):
    a=arms[name];f=a['fit'];g=f['final_support']
    assert a['state_before']==a['state_after']
    assert all(a[k] for k in ('rng_cpu_unchanged','rng_cuda_unchanged','module_modes_unchanged','existing_gradients_unchanged'))
    fields=[f"{100*a['metrics'][k]:.6f}" for k in ('seen','missing','all','macro')]
    fields += [f"{100*g['accuracy']:.6f}",f"{f['before']['ce']:.9g}/{g['ce']:.9g}",f"{g['grad_inf']:.9g}",f"{g['grad_l2']:.9g}",f"{g['weight_norm']:.9g}/{g['bias_norm']:.9g}",f"{f['n_iter']}/{f['func_evals']}"]
    lines.append('| '+name+' | '+' | '.join(fields)+' |')
r=100*paired['metrics']['missing'];s=100*broken['metrics']['missing'];q=r/21.9875;delta=r-s
if paired['fit']['after']['accuracy']<.95: verdict='fit-limited; do not reject representation from weak negative result'
elif q>=.7 and delta>=8: verdict='strong relational-transport gate'
elif q<.4: verdict='simple relation coordinates fail'
else: verdict='ambiguous'
checks.update(R=r,S=s,P=21.9875,q_rel=q,delta_rel=delta,verdict=verdict,
    paired_fit_adequate=paired['fit']['after']['accuracy']>=.95,broken_fit_adequate=broken['fit']['after']['accuracy']>=.95,
    corrected_historical_procrustes_seen=100*historical['result']['metrics']['seen'])
lines+=['',f'R={r:.6f}%,S={s:.6f}%,q_rel={q:.9g},delta_rel={delta:.6f}pp. Frozen branch: {verdict}.',
    f"Historical H04-A N256 Procrustes seen={checks['corrected_historical_procrustes_seen']:.6f}% (corrects44.15 in assignment);missing21.9875%.", '', '| Client | Gram relative disagreement | Gram SHA256 |', '|---|---:|---|']
for i,g in enumerate(paired['gram']): lines.append(f'| {i} | {g["relative_disagreement"]:.9g} | {g["hash"]} |')
lines+=['','| Client | Seed | Fixed points | Permutation SHA256 |','|---|---|---:|---|']
for rec in broken['permutations']:
    p=rec['permutation'];assert sorted(p)==list(range(256))
    assert rec['fixed_points']==sum(j==v for j,v in enumerate(p))
    assert rec['support_test_multisets_bitwise_unchanged']
    assert rec['permutation_sha256']==hashlib.sha256(json.dumps(p,separators=(',',':')).encode()).hexdigest()
    lines.append(f'| {rec["client"]} | {rec["seed"]} | {rec["fixed_points"]} | {rec["permutation_sha256"]} |')
(root/'verification.json').write_text(json.dumps(checks,indent=2))
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
