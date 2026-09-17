import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);checks={};rank_lines=[];hash_lines=[]
lines=['# H04-B cross-seed replication','', '| Seed | Arm | Seen % | Missing % | All % | Macro % | Fit % | CE | grad_inf | grad_l2 | Weight/bias norm | Iter/eval |', '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|']
for seed in (1,2):
    folder=root/f'artifacts/experiment/fedgh_seed{seed}'
    rr=[json.loads(s) for s in (folder/'rounds.jsonl').read_text().splitlines()]
    old=[json.loads(s) for s in Path(f'research_log/H02A/full/artifacts/experiment/fedgh_seed{seed}/rounds.jsonl').read_text().splitlines()]
    assert len(rr)==10
    for r,h in zip(rr,old):
        assert all(r[k]==h[k] for k in ('metrics','client_model_hashes','prototype_bank_hash','server_head'))
    assert all('cross_seed_probe' not in r for r in rr[:9])
    provenance=json.loads((folder/'cross_seed_provenance.json').read_text())
    split=json.loads((folder/'split.json').read_text())
    groups=[set(sum(split['train_indices'],[])),set(provenance['oracle_indices']),set(sum(provenance['support_indices'],[])),set(provenance['anchor_indices'])]
    assert [len(v) for v in groups]==[2000,1000,2000,1000]
    assert all(not a&b for i,a in enumerate(groups) for b in groups[i+1:])
    for name in ('oracle','support','anchor'):
        assert provenance[name+'_sha256']==hashlib.sha256(json.dumps(provenance[name+'_indices'],separators=(',',':')).encode()).hexdigest()
        hash_lines.append(f'| {seed} | {name} | {provenance[name+"_sha256"]} |')
    arms=rr[-1]['cross_seed_probe'];native=arms['native_2000'];adequate=True
    for name,a in arms.items():
        f=a['fit'];g=f['final_support'];adequate &= g['accuracy']>=.95
        assert a['state_before']==a['state_after']==native['state_before']
        assert all(a[k] for k in ('rng_cpu_unchanged','rng_cuda_unchanged','module_modes_unchanged','existing_gradients_unchanged'))
        assert f['max_iter']==2000
        fields=[f"{100*a['metrics'][k]:.6f}" for k in ('seen','missing','all','macro')]
        fields += [f"{100*g['accuracy']:.6f}",f"{g['ce']:.9g}",f"{g['grad_inf']:.9g}",f"{g['grad_l2']:.9g}",f"{g['weight_norm']:.9g}/{g['bias_norm']:.9g}",f"{f['n_iter']}/{f['func_evals']}"]
        lines.append(f'| {seed} | {name} | '+' | '.join(fields)+' |')
        if 'alignment' in a:
            n=1000 if name=='paired_1000_2000' else 256
            assert a['anchor_receipt']['indices']==provenance['anchor_indices'][:n]
            for i,(rank,d) in enumerate(zip(a['rank_diagnostics'],a['alignment'])):
                rank_lines.append(f'| {seed} | {n} | {i} | {rank["effective_rank"]}/{rank["centered_rank_ceiling"]} | {d["centered_residual_before"]:.9g}/{d["centered_residual_after"]:.9g} | {d["relative_residual_reduction"]:.9g} | {d["orthogonality_error_double"]:.9g}/{d["orthogonality_error_applied"]:.9g} | {d["transform_hash"]} |')
    b=100*native['metrics']['missing'];p=100*arms['paired_1000_2000']['metrics']['missing'];s=100*arms['paired_256_2000']['metrics']['missing']
    checks[str(seed)]=dict(B=b,P1000=p,P256=s,G1000=p-b,G256=s-b,Rgain=(s-b)/(p-b) if p>b else None,
        fits_adequate=adequate,all_ten_online_exact=True,disjoint=True,state_rng_modes_gradients_unchanged=True)
values=list(checks.values())
if not all(v['fits_adequate'] for v in values): verdict='fit-limited; formal cross-seed gate unresolved'
elif all(v['G1000']>=10 and v['G256']>=8 and v['Rgain']>=.75 for v in values): verdict='cross-seed mechanism/compression replicated'
elif all(v['G1000']>=10 for v in values) and any(v['Rgain']<.60 for v in values): verdict='mechanism replicates but N256 compression does not'
elif any(v['G1000']<5 for v in values): verdict='mechanism itself fails to replicate'
else: verdict='ambiguous'
lines+=['','| Seed | B | P1000 | P256 | G1000 pp | G256 pp | Rgain |','|---|---:|---:|---:|---:|---:|---:|']
for seed,v in checks.items(): lines.append('| '+seed+' | '+' | '.join(str(v[k]) for k in ('B','P1000','P256','G1000','G256','Rgain'))+' |')
lines+=['',f'Frozen branch: {verdict}.', '', 'N256 payload524288 bytes/client (5242880 total),exact3.90625x reduction from N1000. Diagnostic anchor features only; support side channel excluded,not a deployable protocol.', '', '| Seed | Set | SHA256 |','|---|---|---|']+hash_lines
lines+=['','| Seed | N | Client | Rank/ceiling | Residual before/after | Reduction | Orthogonality double/applied | Transform hash |','|---|---|---|---|---|---|---|---|']+rank_lines
checks['verdict']=verdict
(root/'verification.json').write_text(json.dumps(checks,indent=2))
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines[:22]));print(json.dumps(checks,indent=2))
