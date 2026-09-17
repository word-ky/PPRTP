import json
from pathlib import Path
import sys

root=Path(sys.argv[1])
folder=root/'artifacts/experiment/fedgh_seed0'
rr=[json.loads(s) for s in (folder/'rounds.jsonl').read_text().splitlines()]
old=[json.loads(s) for s in Path('research_log/H02A/full/artifacts/experiment/fedgh_seed0/rounds.jsonl').read_text().splitlines()]
checks=[]
for i,r in enumerate(rr):
    p=r['probe']
    checks.append(dict(round=r['round'],online_exact=all(r[k]==old[i][k] for k in
        ('client_model_hashes','prototype_bank_hash','server_head')) and
        all(r['metrics'][k]==v for k,v in old[i]['metrics'].items()),
        clients_unchanged=p['client_hashes_before']==p['client_hashes_after'],
        server_unchanged=p['server_hash_before']==p['server_hash_after'],
        prototype_fit_adequate=p['after']['accuracy']>=.95))
assert all(c['online_exact'] and c['clients_unchanged'] and c['server_unchanged'] for c in checks)
(root/'verification.json').write_text(json.dumps(checks,indent=2))
lines=['# H02-B server-head adequacy probe','',
       'Seed0 only. All online hashes and metrics exactly reproduce H02-A; probe changes no client or online server parameters.',
       'LBFGS full batch, lr1 (default), max_iter100, strong_wolfe, tolerance_grad1e-9, tolerance_change1e-12; no regularizer.',
       '', '| Round | Readout | Seen % | Missing % | All % | Macro % |','|---|---|---:|---:|---:|---:|']
for r in rr:
    if r['round'] not in (2,10): continue
    for key in ('global_head_post_server','probe_head_postfit'):
        lines.append(f"| {r['round']} | {key} | "+' | '.join(f"{100*r['metrics'][key][m]:.6f}" for m in ('seen','missing','all','macro'))+' |')
lines+=['','| Round | Probe CE before / after | Prototype acc before / after | LBFGS iterations / evaluations | Owner cosine mean / min / max |',
        '|---|---|---|---|---|']
for r in rr:
    p=r['probe']; o=r['owner_prototype_compatibility']
    lines.append(f"| {r['round']} | {p['before']['ce']:.9g} / {p['after']['ce']:.9g} | "
        f"{p['before']['accuracy']:.6f} / {p['after']['accuracy']:.6f} | {p['n_iter']} / {p['func_evals']} | "+
        ' / '.join(f'{o[k]:.6f}' for k in ('mean','min','max'))+' |')
lines+=['','Per-client metrics, norms, hashes and side-effect receipts are preserved in raw JSON. All accuracy numbers use the same official test subset as H02-A.',
        'No explicit optimizer termination reason is exposed by PyTorch; counts are logged. High prototype fit does not establish test generalization.']
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
