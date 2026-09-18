import json,sys,math
from pathlib import Path
root=Path(sys.argv[1]);checks={};details=[];oracle_lines=['','| Seed | Oracle seen % | Oracle missing % | Oracle all % |','|---|---:|---:|---:|']
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
lines=['# H10-A LOO90 native-radius router','','| Seed | Arm | Seen % | Missing % | All % | Macro % | Classes |','|---|---|---:|---:|---:|---:|---:|']
for seed in (0,1,2):
    folder=root/f'artifacts/experiment/fedgh_seed{seed}'
    rr=[json.loads(s) for s in (folder/'rounds.jsonl').read_text().splitlines()]
    old=[json.loads(s) for s in Path(f'research_log/H02A/full/artifacts/experiment/fedgh_seed{seed}/rounds.jsonl').read_text().splitlines()]
    assert len(rr)==len(old)==10
    for r,h in zip(rr,old): assert all(r[k]==h[k] for k in ('metrics','client_model_hashes','prototype_bank_hash','server_head'))
    h=read(f'research_log/H09A/full/artifacts/experiment/fedgh_seed{seed}/final.json')['dual_space_probe']
    assert rr[-1]['dual_space_probe']==h
    p=rr[-1]['radius_router_probe'];assert p['h09a_entire_reference_exact']
    a=h['aligned_global_prototype_cosine'];raw=h['dual_space_owner_seen_aligned_missing'];d=p['loo90_native_accept_else_aligned_missing'];cal=p['calibration']
    for iso in (d['isolation'],cal['isolation']): assert iso['state_before']==iso['state_after']==a['state_before'] and iso['state_rng_modes_gradients_unchanged']
    assert d['global_hash']==a['global_hash'] and d['transform_hashes']==[v['transform_hash'] for v in a['alignment']]
    assert d['radius_hashes']==cal['radius_hashes'] and len(cal['radii'])==20
    assert cal['local_radius_bytes_per_client']==[8]*10 and not cal['formal_conformal_guarantee']
    for v in cal['radii']:
        assert v['n']==100 and v['alpha']==.10 and v['rank_1indexed']==91 and math.isfinite(v['radius'])
        assert v['raw_owner_hash']==next(x['raw_hash'] for x in a['local_prototypes'] if x['client']==v['client'] and x['label']==v['label'])
    for name,arm in [('aligned_global_prototype_cosine',a),('dual_space_owner_seen_aligned_missing',raw),('loo90_native_accept_else_aligned_missing',d)]:
        lines.append(f'| {seed} | {name} | '+' | '.join(f"{100*arm['metrics'][k]:.6f}" for k in ('seen','missing','all','macro'))+f" | {arm['predicted_class_count']} |")
    oracle=d['oracle_seen_missing_router'];om=oracle['metrics']
    assert oracle['diagnostic_only']
    split=read(folder/'split.json')
    for classes,v,c in zip(split['class_sets'],oracle['per_client'],raw['component_diagnostics']['per_client']):
        assert sum(v['class_correct'][k] for k in classes)==c['native_owner_seen_only_correct']
        assert sum(v['class_correct'][k] for k in range(10) if k not in classes)==c['aligned_missing_only_correct']
    oracle_lines.append(f'| {seed} | '+' | '.join(f"{100*om[k]:.6f}" for k in ('seen','missing','all'))+' |')
    delta={k:100*(d['metrics'][k]-a['metrics'][k]) for k in ('seen','missing','all')}
    checks[str(seed)]=dict(h09a_h07_entire_references_exact=True,all_ten_online_exact=True,state_rng_modes_gradients_unchanged=True,oracle_components_exact=True,
        delta_pp=delta,strong=delta['missing']>=-2 and delta['seen']>=15 and delta['all']>=2 and d['predicted_class_count']>=9,
        oracle_all=om['all'],routing=d['routing_diagnostics']['total'])
    details+=['',f'Seed {seed} routing: '+json.dumps(d['routing_diagnostics']['total']),
        'Prediction histograms: '+json.dumps(d['prediction_histograms']['total']),
        '', '| Client | Class | Radius (rank91) | LOO p10 | LOO p50 | LOO p90 | LOO accept | Full-mean own-class train accept | Any-owned train accept |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for v in cal['radii']:
        x=v['loo_nonconformity'];details.append(f"| {v['client']} | {v['label']} | {v['radius']:.9g} | {x['p10']:.9g} | {x['p50']:.9g} | {x['p90']:.9g} | {v['loo_accept_rate']:.6f} | {v['fullmean_selfclass_train_accept_rate']:.6f} | {v['any_owned_train_accept_rate']:.6f} |")
count=sum(v['strong'] for v in checks.values())
verdict='strong' if count==3 else ('intermediate' if count else ('simple radial known-class detection insufficient despite component headroom' if all(v['oracle_all']>.30 for v in checks.values()) else 'oracle headroom mismatch requires diagnosis'))
checks.update(strong_seed_count=count,verdict=verdict)
lines+=['',f'Strong seeds: {count}/3. Frozen verdict: {verdict}.']+oracle_lines+['',
    'Oracle uses true labels only after proposed predictions; it is not a method. Fixed alpha=.10, n100, rank91, no tuning. Radius fitting refreshes2000 ordinary-local examples per seed under final eval/no-grad model.',
    'Incremental communication0B; local storage2float32 scalar radii/client=8B/client,80B across10clients. Prior anchor/global-bank/affine delivery caveats remain. No formal conformal finite-sample guarantee. Full client/class counts, routing confusion and radius hashes are in final.json.']+details
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');(root/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps(checks,indent=2))
