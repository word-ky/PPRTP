import json,sys
from pathlib import Path
root=Path(sys.argv[1]);checks={};details=[]
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
lines=['# H09-B centered residual readout','','| Seed | Arm | Seen % | Missing % | All % | Macro % | Classes |','|---|---|---:|---:|---:|---:|---:|']
for seed in (0,1,2):
    folder=root/f'artifacts/experiment/fedgh_seed{seed}'
    rr=[json.loads(s) for s in (folder/'rounds.jsonl').read_text().splitlines()]
    old=[json.loads(s) for s in Path(f'research_log/H02A/full/artifacts/experiment/fedgh_seed{seed}/rounds.jsonl').read_text().splitlines()]
    assert len(rr)==len(old)==10
    for r,h in zip(rr,old): assert all(r[k]==h[k] for k in ('metrics','client_model_hashes','prototype_bank_hash','server_head'))
    h=read(f'research_log/H09A/full/artifacts/experiment/fedgh_seed{seed}/final.json')['dual_space_probe']
    assert rr[-1]['dual_space_probe']==h
    p=rr[-1]['centered_dual_probe'];assert p['h09a_entire_reference_exact']
    a=h['aligned_global_prototype_cosine'];raw=h['dual_space_owner_seen_aligned_missing']
    d=p['centered_dual_owner_seen_aligned_missing'];g=p['centered_aligned_global_only']
    for arm in (d,g):
        assert arm['state_before']==arm['state_after']==a['state_before'] and arm['state_rng_modes_gradients_unchanged'] and arm['scores_finite']
        assert arm['global_hash']==a['global_hash'] and arm['transform_hashes']==[v['transform_hash'] for v in a['alignment']]
        assert arm['owner_raw_hashes']==raw['owner_raw_hashes']
        for v in arm['centered_diagnostics']:
            assert v['dtype']=='torch.float32' and v['atol']<=2e-5 and v['rtol']<=2e-5
            assert v['norms']['owner_centered']['min']>0 and v['norms']['global_centered']['min']>0
    for name,arm in [('aligned_global_prototype_cosine',a),('dual_space_owner_seen_aligned_missing',raw),('centered_dual_owner_seen_aligned_missing',d),('centered_aligned_global_only',g)]:
        lines.append(f'| {seed} | {name} | '+' | '.join(f"{100*arm['metrics'][k]:.6f}" for k in ('seen','missing','all','macro'))+f" | {arm['predicted_class_count']} |")
    delta={k:100*(d['metrics'][k]-a['metrics'][k]) for k in ('seen','missing','all')}
    checks[str(seed)]=dict(h09a_h07_entire_references_exact=True,all_ten_online_exact=True,state_rng_modes_gradients_unchanged=True,delta_pp=delta,
        strong=delta['missing']>=-2 and delta['seen']>=15 and delta['all']>=2 and d['predicted_class_count']>=9,
        owner_rotation_max_abs_error=d['owner_rotation_max_abs_error'],
        dual_minus_centered_global_pp={k:100*(d['metrics'][k]-g['metrics'][k]) for k in ('seen','missing','all')},
        component_diagnostics={k:d['component_diagnostics'][k] for k in ('native_owner_seen_only','aligned_missing_only')})
    details+=['',f'Seed {seed}: worst owner rotation error={d["owner_rotation_max_abs_error"]:.9g}.',
        'Centered component diagnostics: '+json.dumps(checks[str(seed)]['component_diagnostics']),
        'Winning group fractions: '+json.dumps(d['winning_group_fraction']),
        'Prediction histograms: '+json.dumps(d['prediction_histograms']['total']),
        '', '| Norm object | Minimum over clients | Mean of client means | Maximum over clients |','|---|---:|---:|---:|']
    for key in d['centered_diagnostics'][0]['norms']:
        values=[v['norms'][key] for v in d['centered_diagnostics']]
        details.append(f"| {key} | {min(v['min'] for v in values):.9g} | {sum(v['mean'] for v in values)/len(values):.9g} | {max(v['max'] for v in values):.9g} |")
    details+=['','| True subset | Score | Mean | p10 | p50 | p90 |','|---|---|---:|---:|---:|---:|']
    for subset,v in d['score_distributions']['pooled'].items():
        for score,s in v['scores'].items(): details.append(f'| {subset} | {score} | '+' | '.join(f"{s[k]:.9g}" for k in ('mean','p10','p50','p90'))+' |')
    details+=['']+[f"Native/aligned winning fractions on true-{subset}: {v['native_winner_fraction']:.9g}/{v['aligned_winner_fraction']:.9g}." for subset,v in d['score_distributions']['pooled'].items()]
count=sum(v['strong'] for v in checks.values());checks['strong_seed_count']=count
checks['gate']='strong' if count==3 else ('intermediate' if count else '0/3 pass; inspect component and centered-global control')
lines+=['',f'Strong seeds: {count}/3. '+checks['gate'],
    'Centering uses existing affine means only, no fitting or test-label routing. Component diagnostics are label-partitioned only. All per-client/class counts, norms/quantiles, histograms and hashes are in final.json.',
    'Incremental communication0B relative to H09-A/H07; existing anchor/global-bank/affine delivery caveats remain.']+details
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');(root/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps(checks,indent=2))
