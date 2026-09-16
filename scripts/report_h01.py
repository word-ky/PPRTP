"""Summarize preserved run receipts without selecting or tuning outcomes."""
import argparse
import json
import math
from pathlib import Path
import shutil
import statistics

parser=argparse.ArgumentParser()
parser.add_argument('runs',nargs='+')
parser.add_argument('--output',default='research_log/H01B')
args=parser.parse_args()
output=Path(args.output)
output.mkdir(parents=True,exist_ok=True)
records={}
metadata={}
for run in map(Path,args.runs):
    experiment=run/'artifacts'/'experiment'
    for source in experiment.glob('*_seed*'):
        if not source.is_dir():
            continue
        mode,seed=source.name.split('_seed'); seed=int(seed)
        target=output/'receipts'/source.name
        target.mkdir(parents=True,exist_ok=True)
        for path in source.glob('*.json*'):
            shutil.copy2(path,target/path.name)
        records[mode,seed]=[json.loads(s) for s in (source/'rounds.jsonl').read_text().splitlines()]
        metadata[mode,seed]=json.loads((source/'metadata.json').read_text())
    for path in experiment.glob('round_one_pairing*.json'):
        shutil.copy2(path,output/path.name)
    for name in ('meta.json','train.log'):
        shutil.copy2(run/name,output/f'{run.name}_{name}')
    shutil.copy2(run/'artifacts'/'tests.txt',output/f'{run.name}_tests.txt')
seeds=sorted({key[1] for key in records})
modes=['local','fedproto','gpc']
checks={}
for seed in seeds:
    first=[records[mode,seed][0] for mode in modes]
    checks[seed]=dict(round_one_equal=all(
        r['client_model_hashes']==first[0]['client_model_hashes'] and r['prototype_bank_hash']==first[0]['prototype_bank_hash']
        for r in first),same_initial=len({metadata[mode,seed]['initial_state_sha256'] for mode in modes})==1,
        same_split=len({metadata[mode,seed]['split_sha256'] for mode in modes})==1,
        round_two_distinct=len({records[mode,seed][1]['prototype_bank_hash'] for mode in modes})==3)
    checks[seed]['finite_losses_and_prototypes']=all(
        math.isfinite(n) and n>0 for mode in modes for r in records[mode,seed] for n in r['prototype_norms']) and all(
        math.isfinite(v) for mode in modes for r in records[mode,seed] for loss in r['losses'] for v in loss.values())
assert all(all(v.values()) for v in checks.values()),checks
(output/'verification.json').write_text(json.dumps(checks,indent=2))
def stat(values):
    avg=statistics.mean(values)
    sd=statistics.stdev(values) if len(values)>1 else 0.
    return f'{100*avg:.2f} ± {100*sd:.2f}'
lines=['# H01-B results','',f'Seeds: {seeds}. Accuracy percentages; mean ± sample standard deviation across seeds.',
       'CIFAR-10 subset: 2000 train examples, 1000 official test examples; 10 clients, K=2.',
       'Frozen configuration: 10 rounds, 1 epoch/round, SGD .01, lambda 1, scale 10.',
       f"Source: `{next(iter(metadata.values()))['source_sha']}`.",'',
       'All seeds passed exact round-1 client/prototype equality and matched initial weights/splits.',
       'All three round-2 prototype banks differ: identical-looking accuracies do not imply an inactive loss.','',
       '| Round | Training | Readout | Seen | Missing | All | Macro |',
       '|---|---|---|---:|---:|---:|---:|']
for r in (2,10):
    for mode in modes:
        for readout in ('head','cosine','l2'):
            values=[stat([records[mode,s][r-1]['metrics'][readout][metric] for s in seeds])
                    for metric in ('seen','missing','all','macro')]
            lines.append(f'| {r} | {mode} | {readout} | '+' | '.join(values)+' |')
lines+=['','## First active knowledge round: client 0, first batch','',
        '| Seed | Training | Local grad norm | Lambda-scaled knowledge norm | Ratio | Missing probability |',
        '|---|---|---:|---:|---:|---:|']
for seed in seeds:
    for mode in ('fedproto','gpc'):
        d=records[mode,seed][1]['diagnostic_client0']
        vals=[f"{d[k]:.6g}" for k in ('local_grad_norm','scaled_knowledge_grad_norm','knowledge_local_grad_ratio','missing_probability_on_seen')]
        lines.append(f'| {seed} | {mode} | '+' | '.join(vals)+' |')
lines+=['','## Runtime and prototype norms','',
        '| Seed | Training | Training/evaluation seconds | Prototype norm min/max over rounds | Final off-diagonal cosine min/max |',
        '|---|---|---:|---:|---:|']
for seed in seeds:
    for mode in modes:
        rr=records[mode,seed]
        norms=[n for r in rr for n in r['prototype_norms']]
        off=[rr[-1]['prototype_cosine'][i][j] for i in range(10) for j in range(10) if i!=j]
        lines.append(f"| {seed} | {mode} | {rr[-1]['elapsed_seconds']:.2f} | {min(norms):.6g} / {max(norms):.6g} | {min(off):.6g} / {max(off):.6g} |")
lines+=['','Local head is the local-only deployment result. Local cosine/L2 are posthoc prototype probes requiring communication.',
        'Primary injection contrast compares GPC and FedProto under the SAME cosine readout.',
        'Macro equals all-class accuracy here because the test set is class-balanced; client metrics are averaged equally.',
        'Common payload per round: 40,960 upload vector bytes + 160 count bytes; 204,800 download vector bytes.',
        'No serialization or class-ID overhead included. All ten prototypes valid after round 1; round 1 training has no knowledge loss.',
        'Per-client counts, metrics, hashes, loss traces and cosine matrices are retained in receipts/.',
        'No claim of convergence or benchmark-scale performance follows from this short subset run.']
(output/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
