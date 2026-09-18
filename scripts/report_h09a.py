import json,sys
from pathlib import Path
root=Path(sys.argv[1]);checks={};details=[]
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
lines=['# H09-A dual-space classifier','','| Seed | Arm | Seen % | Missing % | All % | Macro % | Classes |','|---|---|---:|---:|---:|---:|---:|']
for seed in (0,1,2):
    folder=root/f'artifacts/experiment/fedgh_seed{seed}'
    rr=[json.loads(s) for s in (folder/'rounds.jsonl').read_text().splitlines()]
    old=[json.loads(s) for s in Path(f'research_log/H02A/full/artifacts/experiment/fedgh_seed{seed}/rounds.jsonl').read_text().splitlines()]
    assert len(rr)==len(old)==10
    for r,h in zip(rr,old): assert all(r[k]==h[k] for k in ('metrics','client_model_hashes','prototype_bank_hash','server_head'))
    p=rr[-1]['dual_space_probe'];assert p['h07_entire_references_exact']
    a=p['aligned_global_prototype_cosine'];n=p['native_global_prototype_cosine_control'];d=p['dual_space_owner_seen_aligned_missing']
    if seed==0:
        h=read('research_log/H07A/gate/artifacts/experiment/fedgh_seed0/final.json')['local_source_probe']
        assert a==h['localtrain_aligned_global_prototype_cosine'] and n==h['localtrain_native_global_prototype_cosine_control']
    else:
        h=read(f'research_log/H07B/full/artifacts/experiment/fedgh_seed{seed}/final.json')['direct_cross_seed_probe']['arms']
        assert a==h[f'seed{seed}_localtrain_aligned_global_prototype_cosine'] and n==h[f'seed{seed}_localtrain_native_global_prototype_cosine_control']
    assert d['state_before']==d['state_after']==a['state_before'] and d['state_rng_modes_gradients_unchanged'] and d['scores_finite']
    assert d['global_hash']==a['global_hash'] and d['transform_hashes']==[v['transform_hash'] for v in a['alignment']]
    for i,hashes in enumerate(d['owner_raw_hashes']): assert hashes==[v['raw_hash'] for v in a['local_prototypes'] if v['client']==i]
    for name in ('aligned_global_prototype_cosine','native_global_prototype_cosine_control','dual_space_owner_seen_aligned_missing'):
        arm=p[name];lines.append(f'| {seed} | {name} | '+' | '.join(f"{100*arm['metrics'][k]:.6f}" for k in ('seen','missing','all','macro'))+f" | {arm['predicted_class_count']} |")
    delta={k:100*(d['metrics'][k]-a['metrics'][k]) for k in ('seen','missing','all')}
    strong=delta['missing']>=-2 and delta['seen']>=15 and delta['all']>=2 and d['predicted_class_count']>=9
    c=d['component_diagnostics']
    checks[str(seed)]=dict(h07_entire_references_exact=True,all_ten_online_exact=True,state_rng_modes_gradients_unchanged=True,delta_pp=delta,strong=strong,
        native_owner_seen_only=c['native_owner_seen_only'],aligned_missing_only=c['aligned_missing_only'],
        component_gain_pp={k:100*c[field]-100*a['metrics'][k] for k,field in [('seen','native_owner_seen_only'),('missing','aligned_missing_only')]})
    details+=['',f"Seed {seed} component diagnostics: native_owner_seen_only={100*c['native_owner_seen_only']:.6f}%; aligned_missing_only={100*c['aligned_missing_only']:.6f}%.",
        'Winning group fractions: '+json.dumps(d['winning_group_fraction']),'Prediction histograms: '+json.dumps(d['prediction_histograms']['total']),
        'Global bank SHA256: '+d['global_hash'],'','| True subset | Score | Mean | p10 | p50 | p90 |','|---|---|---:|---:|---:|---:|']
    for subset,v in d['score_distributions']['pooled'].items():
        for score,s in v['scores'].items(): details.append(f'| {subset} | {score} | '+' | '.join(f"{s[k]:.9g}" for k in ('mean','p10','p50','p90'))+' |')
    details+=['']+[f"Native/aligned winning fractions on true-{subset}: {v['native_winner_fraction']:.9g}/{v['aligned_winner_fraction']:.9g}." for subset,v in d['score_distributions']['pooled'].items()]
count=sum(v['strong'] for v in checks.values());checks['strong_seed_count']=count
checks['gate']='strong' if count==3 else ('intermediate' if count else 'no seed passes; inspect components and score dominance')
lines+=['',f'Strong seeds: {count}/3. '+checks['gate'],
    'Components are label-partitioned diagnostics only; proposed scores/predictions never use test labels. Full per-client/class counts, histograms and score distributions are in final.json.',
    'Incremental communication0B relative to H07 aligned inference. Owner prototypes stay local. Prior anchor/global-bank/affine costs and incomplete distributed-protocol caveat remain.']+details
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');(root/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps(checks,indent=2))
