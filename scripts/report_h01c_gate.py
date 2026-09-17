"""Report the predeclared first-seed strength check; never tune lambda."""
import json
from pathlib import Path
import shutil
import sys

run=Path(sys.argv[1])
out=Path('research_log/H01C')
out.mkdir(parents=True,exist_ok=True)
modes=['fedproto','gpc_all_match','gpc_seen_match']
rr={}
for mode in modes:
    src=run/'artifacts/experiment'/f'{mode}_seed0'
    target=out/'receipts'/src.name
    target.mkdir(parents=True,exist_ok=True)
    for path in src.glob('*.json*'):
        shutil.copy2(path,target/path.name)
    rr[mode]=[json.loads(s) for s in (src/'rounds.jsonl').read_text().splitlines()]
for name in ('meta.json','train.log'):
    shutil.copy2(run/name,out/name)
shutil.copy2(run/'artifacts/tests.txt',out/'tests.txt')
shutil.copy2(run/'artifacts/experiment/round_one_pairing_seed0.json',out/'pairing.json')
ref=rr['fedproto'][1]['diagnostic_client0']['scaled_knowledge_grad_norm']
ratios={m:rr[m][1]['diagnostic_client0']['scaled_knowledge_grad_norm']/ref for m in modes[1:]}
passed=all(.5<=v<=2 for v in ratios.values())
baseline=Path('research_log/H01B/receipts/fedproto_seed0/rounds.jsonl')
old=[json.loads(s) for s in baseline.read_text().splitlines()]
parity=all(rr['fedproto'][i]['client_model_hashes']==old[i]['client_model_hashes'] and
           rr['fedproto'][i]['prototype_bank_hash']==old[i]['prototype_bank_hash'] for i in (0,1))
receipt=dict(passed=passed,window=[.5,2],ratios_to_fedproto=ratios,baseline_hash_parity=parity,
    decision='continue frozen matrix' if passed else 'STOP; no tuning; later rounds/seeds NOT RUN')
(out/'gate.json').write_text(json.dumps(receipt,indent=2))
lines=['# H01-C seed-0 strength gate','',f'Gate: {receipt["decision"]}. H01-B FedProto first-two-round hash parity: {parity}.','',
       '| Mode | Local grad | Scaled knowledge grad | Knowledge/local | Knowledge/FedProto |',
       '|---|---:|---:|---:|---:|']
for mode in modes:
    d=rr[mode][1]['diagnostic_client0']
    lines.append(f"| {mode} | {d['local_grad_norm']:.9g} | {d['scaled_knowledge_grad_norm']:.9g} | {d['knowledge_local_grad_ratio']:.9g} | {d['scaled_knowledge_grad_norm']/ref:.9g} |")
lines+=['','Round 2 accuracy percentages:','', '| Mode | Readout | Seen | Missing | All | Macro |','|---|---|---:|---:|---:|---:|']
for mode in modes:
    for readout,v in rr[mode][1]['metrics'].items():
        lines.append(f'| {mode} | {readout} | '+' | '.join(f'{100*v[k]:.4f}' for k in ('seen','missing','all','macro'))+' |')
lines+=['','Cross-client same-class prototype cosine, computed before aggregation:','',
        '| Round | Mode | Mean | Min | Max |','|---|---|---:|---:|---:|']
for i in (0,1):
    for mode in modes:
        d=rr[mode][i]['owner_prototype_compatibility']
        lines.append(f'| {i+1} | {mode} | '+ ' | '.join(f'{d[k]:.9g}' for k in ('mean','min','max'))+' |')
lines+=['','Round-1 pairing checked across all three arms; exact same initial states, data and batch order.',
        'This short gate is not a round-10 conclusion. Round-2 metrics are descriptive only.']
(out/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(receipt,indent=2))
print('\n'.join(lines))
