"""Summarize H01-D gate/full receipts at fixed settings."""
import argparse
import json
import math
from pathlib import Path
import shutil
import statistics

p=argparse.ArgumentParser()
p.add_argument('run',type=Path)
p.add_argument('--gate',action='store_true')
a=p.parse_args()
out=Path('research_log/H01D')/('gate' if a.gate else 'full')
out.mkdir(parents=True,exist_ok=True)
modes=['fedproto','gpc_all_match','gpc_seen_match']
rr={}; meta={}
for src in (a.run/'artifacts/experiment').glob('*_seed*'):
    if not src.is_dir():
        continue
    mode,seed=src.name.split('_seed'); seed=int(seed)
    target=out/'receipts'/src.name; target.mkdir(parents=True,exist_ok=True)
    for path in src.glob('*.json*'):
        shutil.copy2(path,target/path.name)
    rr[mode,seed]=[json.loads(s) for s in (src/'rounds.jsonl').read_text().splitlines()]
    meta[mode,seed]=json.loads((src/'metadata.json').read_text())
for name in ('meta.json','train.log'):
    shutil.copy2(a.run/name,out/name)
shutil.copy2(a.run/'artifacts/tests.txt',out/'tests.txt')
for path in (a.run/'artifacts/experiment').glob('round_one_pairing*.json'):
    shutil.copy2(path,out/path.name)
seeds=sorted({k[1] for k in rr})
checks={}
for seed in seeds:
    first=rr[modes[0],seed][0]
    checks[seed]=dict(paired=all(rr[m,seed][0]['client_model_hashes']==first['client_model_hashes'] and
        rr[m,seed][0]['prototype_bank_hash']==first['prototype_bank_hash'] for m in modes),
        same_split=len({meta[m,seed]['split_sha256'] for m in modes})==1,
        same_initial=len({meta[m,seed]['initial_state_sha256'] for m in modes})==1,
        finite=all(math.isfinite(n) and n>0 for m in modes for r in rr[m,seed] for n in r['prototype_norms']))
    old=[json.loads(s) for s in (Path('research_log/H01B/receipts')/f'fedproto_seed{seed}'/'rounds.jsonl').read_text().splitlines()]
    checks[seed]['fedproto_baseline_parity']=all(r['client_model_hashes']==old[i]['client_model_hashes'] and
        r['prototype_bank_hash']==old[i]['prototype_bank_hash'] for i,r in enumerate(rr['fedproto',seed]))
assert all(all(v.values()) for v in checks.values()),checks
def diag(mode,seed,r):
    return rr[mode,seed][r-1]['diagnostic_client0']
ratio=diag('gpc_seen_match',0,2)['scaled_knowledge_grad_norm']/diag('gpc_all_match',0,2)['scaled_knowledge_grad_norm']
verification=dict(checks=checks,seed0_round2_seen_all=ratio,gate_passed=.8<=ratio<=1.25)
(out/'verification.json').write_text(json.dumps(verification,indent=2))
lines=['# H01-D '+('gate' if a.gate else 'full result'),'',
    f'Source `{meta[modes[0],0]["source_sha"]}`. Seeds {seeds}. Seed0 round2 seen/all ratio {ratio:.9g}; gate {.8<=ratio<=1.25}.',
    'Exact first-round pairing and FedProto historical model/prototype hash parity passed.','',
    'Accuracies in percent, mean ± sample SD. Frozen lambdas: FedProto 1, all .002, seen .03498.','',
    '| Round | Mode | Readout | Seen | Missing | All | Macro |','|---|---|---|---:|---:|---:|---:|']
for r in ([2] if a.gate else [2,10]):
    for m in modes:
        for readout in ('cosine','head','l2'):
            values=[]
            for metric in ('seen','missing','all','macro'):
                v=[100*rr[m,s][r-1]['metrics'][readout][metric] for s in seeds]
                values.append(f'{statistics.mean(v):.4f} ± {statistics.stdev(v) if len(v)>1 else 0:.4f}')
            lines.append(f'| {r} | {m} | {readout} | '+' | '.join(values)+' |')
lines+=['','## Same feature / same bank gradient direction at seed0 round2','',
    '| Arm | Feature-gradient cosine | Unscaled all/seen feature-gradient norm ratio | Base local norm | Base scaled knowledge norm |',
    '|---|---:|---:|---:|---:|']
for m in modes:
    d=diag(m,0,2); f=d['denominator_feature_gradients']
    lines.append(f"| {m} | {f['cosine']:.9g} | {f['unscaled_all_seen_norm_ratio']:.9g} | {d['local_grad_norm']:.9g} | {d['scaled_knowledge_grad_norm']:.9g} |")
lines+=['','## Strength ratio seen/all across respective training arms','',
    '| Seed | Round | Scaled base-gradient ratio |','|---|---|---:|']
for s in seeds:
    for r in ([2] if a.gate else [2,5,10]):
        ratio=diag('gpc_seen_match',s,r)['scaled_knowledge_grad_norm']/diag('gpc_all_match',s,r)['scaled_knowledge_grad_norm']
        lines.append(f'| {s} | {r} | {ratio:.9g} |')
lines+=['','## Owner prototype compatibility (before aggregation)','',
    '| Seed | Round | Arm | Mean | Min | Max |','|---|---|---|---:|---:|---:|']
for s in seeds:
    for r in ([1,2] if a.gate else [1,2,5,10]):
        for m in modes:
            d=rr[m,s][r-1]['owner_prototype_compatibility']
            lines.append(f'| {s} | {r} | {m} | '+' | '.join(f'{d[k]:.9g}' for k in ('mean','min','max'))+' |')
lines+=['','No retuning after the seed0 gate. Later strength ratios compare diverged arm states, not identical tensors.',
        'Same-tensor direction uses dL/dz; strength calibration uses gradients into base parameters. These norms differ.',
        'Finite prototype norms do not establish convergence. This is a small CIFAR-10 subset experiment.']
(out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(verification,indent=2))
print('\n'.join(lines[:40]))
