import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); folder=root/'artifacts/experiment/fedgh_seed0'
rr=[json.loads(s) for s in (folder/'rounds.jsonl').read_text().splitlines()]
old=[json.loads(s) for s in Path('research_log/H02A/full/artifacts/experiment/fedgh_seed0/rounds.jsonl').read_text().splitlines()]
p=json.loads((folder/'heldout_owner_support.json').read_text()); split=json.loads((folder/'split.json').read_text())
oracle=json.loads(Path('research_log/H02C/full/artifacts/experiment/fedgh_seed0/oracle_calibration.json').read_text())
assert not set(sum(p['indices'],[])).intersection(sum(split['train_indices'],[]))
assert len(set(sum(p['indices'],[])))==2000
assert not set(sum(p['indices'],[])).intersection(oracle['indices'])
assert p['class_counts']==[200]*10 and p['samples_per_client']==[200]*10 and p['total']==2000
assert p['indices_sha256']==hashlib.sha256(json.dumps(p['indices'],separators=(',',':')).encode()).hexdigest()
checks=[]
for r,h in zip(rr,old):
    assert all(r[k]==h[k] for k in ('metrics','client_model_hashes','prototype_bank_hash','server_head'))
    d=dict(round=r['round'],online_exact=True)
    if 'heldout_owner_probe' in r:
        o=r['heldout_owner_probe']
        assert o['state_before']==o['state_after'] and o['rng_cpu_unchanged'] and o['rng_cuda_unchanged']
        assert o['module_modes_unchanged']
        d.update(state_unchanged=True,rng_unchanged=True,module_modes_unchanged=True)
    checks.append(d)
assert len(rr)==10 and [r['round'] for r in rr if 'heldout_owner_probe' in r]==[2,10]
q=100*rr[9]['heldout_owner_probe']['metrics']['missing']/32.725
(root/'verification.json').write_text(json.dumps(dict(rounds=checks,indices_hash=p['indices_sha256'],q=q),indent=2))
lines=['# H02-E held-out owner-support diagnostic','',f"Disjoint held-out indices hash: `{p['indices_sha256']}`; 200/client,2000total,200/class; zero train/oracle/cross-client overlap.",
       'All10round H02-A online records exact; probe preserves client/server/prototype state and CPU/CUDA RNG.',
       '', '| Round | Seen % | Missing % | All % | Macro % | Fit CE before / after | Fit accuracy before / after | LBFGS iterations / evaluations |',
       '|---|---:|---:|---:|---:|---|---|---|']
for r in rr:
    if 'heldout_owner_probe' not in r: continue
    o=r['heldout_owner_probe']; f=o['fit']
    fields=[f"{100*o['metrics'][k]:.6f}" for k in ('seen','missing','all','macro')]
    fields += [f"{f['before']['ce']:.9g} / {f['after']['ce']:.9g}",f"{f['before']['accuracy']:.6f} / {f['after']['accuracy']:.6f}",f"{f['n_iter']} / {f['func_evals']}"]
    lines.append(f"| {r['round']} | "+' | '.join(fields)+' |')
lines+=['',f'Round10 recovered oracle gap q_hold={q:.9g}. Fixed references: reused-owner missing round2=.2125%,round10=0%; shared oracle round2=31.45%,round10=32.725%.',
        'Raw JSON retains all per-client metrics, optimizer diagnostics, parameter hashes/norms and before/after state receipts. This is an analysis-only communication upper bound.']
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))

