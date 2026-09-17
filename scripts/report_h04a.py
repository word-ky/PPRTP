import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);folder=root/'artifacts/experiment/fedgh_seed0'
rr=[json.loads(s) for s in (folder/'rounds.jsonl').read_text().splitlines()]
old=[json.loads(s) for s in Path('research_log/H02A/full/artifacts/experiment/fedgh_seed0/rounds.jsonl').read_text().splitlines()]
assert len(rr)==10
for r,h in zip(rr,old):
    assert all(r[k]==h[k] for k in ('metrics','client_model_hashes','prototype_bank_hash','server_head'))
assert all('anchor_count_probe' not in r for r in rr[:9])
provenance=json.loads((folder/'paired_anchors.json').read_text())
assert provenance==json.loads(Path('research_log/H03D/gate/artifacts/experiment/fedgh_seed0/paired_anchors.json').read_text())
historical=json.loads(Path('research_log/H03D/gate/artifacts/experiment/fedgh_seed0/final.json').read_text())['convexity_probe']['paired_2000']
arms=rr[-1]['anchor_count_probe']['arms']
assert list(arms)==['1000','512','256','128','64']
baseline={k:v for k,v in arms['1000']['result'].items() if k not in ('rank_diagnostics','existing_gradients_unchanged')}
assert baseline==historical
checks=dict(all_ten_online_rounds_exact=True,n1000_exact=True,provenance_exact=True,arms={})
lines=['# H04-A fixed nested anchor counts','', 'Exact H03-D N1000 reproduction. All10 online records, states/RNG/modes/existing gradients unchanged.', '', '| N | Seen % | Missing % | All % | Macro % | Fit % | CE | grad_inf | grad_l2 | Iter/eval | q | Retention |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|']
for key,arm in arms.items():
    n=int(key);a=arm['result'];f=a['fit'];g=f['final_support'];receipt=arm['anchor_receipt']
    assert receipt['indices']==provenance['indices'][:n]
    assert receipt['indices_sha256']==hashlib.sha256(json.dumps(receipt['indices'],separators=(',',':')).encode()).hexdigest()
    assert a['state_before']==a['state_after']==historical['state_before']
    assert all(a[k] for k in ('rng_cpu_unchanged','rng_cuda_unchanged','module_modes_unchanged','existing_gradients_unchanged'))
    assert f['max_iter']==2000
    assert arm['payload_bytes_per_client']==n*512*4 and arm['payload_bytes_total']==10*n*512*4
    p=100*a['metrics']['missing'];q=p/32.725;ret=p/23.55
    checks['arms'][key]=dict(q=q,retention=ret,fit_adequate=g['accuracy']>=.95,missing=p)
    fields=[f"{100*a['metrics'][k]:.6f}" for k in ('seen','missing','all','macro')]
    fields += [f"{100*g['accuracy']:.6f}",f"{g['ce']:.9g}",f"{g['grad_inf']:.9g}",f"{g['grad_l2']:.9g}",f"{f['n_iter']}/{f['func_evals']}",f'{q:.9g}',f'{ret:.9g}']
    lines.append('| '+key+' | '+' | '.join(fields)+' |')
c=checks['arms'];a=c['256'];b=c['512']
if a['fit_adequate'] and a['q']>=.5 and a['retention']>=.8: verdict='strong count compression supported'
elif b['retention']>=.8 and a['retention']<.6: verdict='only mild compression supported (fit-limited caveat applies if relevant)'
elif b['retention']<.6 and b['fit_adequate']: verdict='mechanism fragile even before severe rank deficiency'
else: verdict='intermediate / no decisive frozen branch'
checks['verdict']=verdict
lines+=['',f'Frozen branch: {verdict}. Fit below95% is flagged fit-limited, without solver changes.', '', '| N | Bytes/client | Bytes total | Compression factor | Weight norm | Bias norm | Prefix SHA256 |','|---|---:|---:|---:|---:|---:|---|']
for key,arm in arms.items():
    f=arm['result']['fit']
    lines.append(f'| {key} | {arm["payload_bytes_per_client"]} | {arm["payload_bytes_total"]} | {arm["compression_factor"]} | {f["weight_norm"]:.9g} | {f["bias_norm"]:.9g} | {arm["anchor_receipt"]["indices_sha256"]} |')
lines+=['','Payload counts float32 anchor features only; unchanged owner-support labels/features are excluded. This is an upper-bound diagnostic communication model, not a deployable protocol. 1000/256=3.90625, approximately4x rather than literally at least4x.', '', '| N | Client | Rank ceiling | Effective rank | Tolerance | Largest singular | Smallest nonzero | Centered before | After | Reduction | Orthogonality double/applied | Transform SHA256 |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|']
for key,arm in arms.items():
    a=arm['result']
    for i,(rank,d) in enumerate(zip(a['rank_diagnostics'],a['alignment'])):
        assert rank['centered_rank_ceiling']==min(512,int(key)-1)
        fields=[str(rank[k]) for k in ('centered_rank_ceiling','effective_rank')]
        fields += [f'{rank[k]:.9g}' for k in ('tolerance','largest_singular_value','smallest_nonzero_singular_value')]
        fields += [f'{d[k]:.9g}' for k in ('centered_residual_before','centered_residual_after','relative_residual_reduction')]
        fields += [f'{d["orthogonality_error_double"]:.9g}/{d["orthogonality_error_applied"]:.9g}',d['transform_hash']]
        lines.append(f'| {key} | {i} | '+' | '.join(fields)+' |')
lines+=['','Rank tolerance fixed before execution: feature_dim * float64_epsilon * largest_singular_value of centered cross-covariance. No rank truncation or change to SVD rotation.']
(root/'verification.json').write_text(json.dumps(checks,indent=2))
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines[:15]));print(json.dumps(checks,indent=2))
