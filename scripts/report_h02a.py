"""Report fetched H02-A JSON/logs, without duplicating bulky receipts."""
import json
from pathlib import Path
import statistics
import sys

root=Path(sys.argv[1])
records={}
checks={}
for folder in sorted((root/'artifacts/experiment').glob('fedgh_seed*')):
    seed=int(folder.name.split('seed')[1])
    rr=[json.loads(s) for s in (folder/'rounds.jsonl').read_text().splitlines()]
    records[seed]=rr
    old=json.loads((Path('research_log/H01B/receipts')/f'fedproto_seed{seed}'/'rounds.jsonl').read_text().splitlines()[0])
    checks[seed]=dict(historical_pairing=rr[0]['client_model_hashes']==old['client_model_hashes'] and
        rr[0]['prototype_bank_hash']==old['prototype_bank_hash'],
        server_updated=all(r['server_head']['hash_before']!=r['server_head']['hash_after'] for r in rr),
        server_preserved_bases=all(r['server_head']['bases_unchanged'] for r in rr),
        broadcast_exact=all(r['broadcast']['server_head_hash']==rr[i-1]['server_head']['hash_after'] and
            all(h==r['broadcast']['server_head_hash'] for h in r['broadcast']['client_head_hashes']) for i,r in enumerate(rr) if i),
        broadcast_preserved_personalized_bases=all(r['broadcast']['bases_unchanged'] and
            len(set(r['broadcast']['base_hashes']))==10 for r in rr[1:]),
        persistent_server=all(r['server_head']['hash_before']==rr[i-1]['server_head']['hash_after'] for i,r in enumerate(rr) if i))
assert all(all(v.values()) for v in checks.values()),checks
(root/'verification.json').write_text(json.dumps(checks,indent=2))
lines=['# H02-A FedGH control','', 'All historical pairing, server update, base preservation, broadcast, and persistence checks passed.',
       'Server SGD lr=.01, no momentum/decay; batch size 1; exactly one pass per round in client ID then class ID order.',
       'Full model checkpoints remain in the remote run directory. These receipts retain JSON and logs.',
       'Accuracies: percent, mean ± sample SD.','',
       '| Round | Readout | Seen | Missing | All | Macro |','|---|---|---:|---:|---:|---:|']
for rnd in ([2,10] if len(records[0])==10 else [2]):
    for readout in ('local_head_pre_server','global_head_post_server','cosine','l2'):
        fields=[]
        for k in ('seen','missing','all','macro'):
            values=[100*rr[rnd-1]['metrics'][readout][k] for rr in records.values()]
            fields.append(f'{statistics.mean(values):.4f} ± {statistics.stdev(values) if len(values)>1 else 0:.4f}')
        lines.append(f'| {rnd} | {readout} | '+' | '.join(fields)+' |')
lines+=['','| Seed | Round | Server CE before / after | Server accuracy before / after | Owner cosine mean / min / max |',
        '|---|---|---|---|---|']
for seed,rr in records.items():
    for r in rr:
        d=r['server_head']; owner=r['owner_prototype_compatibility']
        lines.append(f"| {seed} | {r['round']} | {d['before']['ce']:.6f} / {d['after']['ce']:.6f} | "
            f"{d['before']['accuracy']:.4f} / {d['after']['accuracy']:.4f} | "+' / '.join(f'{owner[k]:.6f}' for k in ('mean','min','max'))+' |')
lines+=['','All parameter hashes, deterministic sample orders, per-client readouts, payload sizes and model metadata are in artifacts/experiment.',
        'Comparison to H01-D is a mechanism/method control: FedGH learns a head and retains separate owner means; it is not a same-information causal ablation.']
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(checks))
print('\n'.join(lines[:20]))
