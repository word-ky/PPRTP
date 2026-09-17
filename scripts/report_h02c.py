import hashlib,json,statistics,sys
from pathlib import Path

root=Path(sys.argv[1]); folder=root/'artifacts/experiment/fedgh_seed0'
rr=[json.loads(s) for s in (folder/'rounds.jsonl').read_text().splitlines()]
old=[json.loads(s) for s in Path('research_log/H02A/full/artifacts/experiment/fedgh_seed0/rounds.jsonl').read_text().splitlines()]
cal=json.loads((folder/'oracle_calibration.json').read_text())
split=json.loads((folder/'split.json').read_text())
assert not set(cal['indices']).intersection(i for ii in split['train_indices'] for i in ii)
assert cal['index_sha256']==hashlib.sha256(json.dumps(cal['indices'],separators=(',',':')).encode()).hexdigest()
assert cal['class_counts']==[100]*10 and not cal['test_used_for_fitting']
checks=[]
for r,h in zip(rr,old):
    exact=all(r[k]==h[k] for k in ('metrics','server_head','client_model_hashes','prototype_bank_hash'))
    assert exact
    check=dict(round=r['round'],online_exact=exact)
    if 'oracle' in r:
        o=r['oracle']
        assert o['state_before']==o['state_after'] and o['rng_cpu_unchanged'] and o['rng_cuda_unchanged']
        check.update(state_unchanged=True,rng_unchanged=True)
    checks.append(check)
assert len(rr)==10 and [r['round'] for r in rr if 'oracle' in r]==[1,2,10]
(root/'verification.json').write_text(json.dumps(dict(calibration_hash=cal['index_sha256'],disjoint=True,rounds=checks),indent=2))
lines=['# H02-C oracle diagnostic','',f"Calibration hash `{cal['index_sha256']}`. Official training split only,100/class; no client-training overlap. Seed314159.",
       'Seed0 only; all10 online rounds exactly reproduce H02-A. Oracle state and CPU/CUDA RNG unchanged.',
       'Oracle labels are analysis-only and unavailable to online FL. Settings fixed before observing results.','',
       '| Round | Readout | Seen % | Missing % | All % | Macro % |','|---|---|---:|---:|---:|---:|']
for r in rr:
    if 'oracle' not in r: continue
    for key,m in r['oracle']['metrics'].items():
        lines.append(f"| {r['round']} | {key} | "+' | '.join(f'{100*m[k]:.5f}' for k in ('seen','missing','all','macro'))+' |')
lines+=['','| Round | Individual fit accuracy mean/min/max | Individual fit CE mean/min/max | Shared fit accuracy / CE | Owner cosine mean/min/max |',
        '|---|---|---|---|---|']
for r in rr:
    if 'oracle' not in r: continue
    o=r['oracle']; fields=[]
    for key in ('accuracy','ce'):
        v=[f['after'][key] for f in o['individual_fits']]
        fields.append(' / '.join(f'{x:.7g}' for x in (statistics.mean(v),min(v),max(v))))
    f=o['shared_fit']['after']; fields.append(f"{f['accuracy']:.7g} / {f['ce']:.7g}")
    fields.append(' / '.join(f"{r['owner_prototype_compatibility'][k]:.7g}" for k in ('mean','min','max')))
    lines.append(f"| {r['round']} | "+' | '.join(fields)+' |')
lines+=['','H02-B20-mean probe missing reference: round2=.0125%, round10=0%; reused, not rerun.',
        'Raw rounds.jsonl contains all per-client metrics, before/after fits, head norms/hashes, LBFGS counts and state receipts.']
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
