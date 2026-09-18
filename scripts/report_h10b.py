import json,sys,math
from pathlib import Path
root=Path(sys.argv[1]);checks={};details=[]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
lines=['# H10-B aligned group gate / native owner refinement','','| Seed | Arm | Seen % | Missing % | All % | Macro % | Classes |','|---|---|---:|---:|---:|---:|---:|']
for seed in (0,1,2):
 folder=root/f'artifacts/experiment/fedgh_seed{seed}'
 rr=[json.loads(s) for s in (folder/'rounds.jsonl').read_text().splitlines()]
 old=[json.loads(s) for s in Path(f'research_log/H02A/full/artifacts/experiment/fedgh_seed{seed}/rounds.jsonl').read_text().splitlines()]
 assert len(rr)==len(old)==10
 for r,h in zip(rr,old):assert all(r[k]==h[k] for k in ('metrics','client_model_hashes','prototype_bank_hash','server_head'))
 h=read(f'research_log/H09A/full/artifacts/experiment/fedgh_seed{seed}/final.json')['dual_space_probe'];assert rr[-1]['dual_space_probe']==h
 p=rr[-1]['group_refine_probe'];assert p['h09a_entire_reference_exact']
 a=h['aligned_global_prototype_cosine'];raw=h['dual_space_owner_seen_aligned_missing'];d=p['aligned_group_gate_native_owner_refine']
 assert d['shared_per_client']==a['per_client'] and d['missing_correctness_per_example_exact']
 iso=d['isolation'];assert iso['state_before']==iso['state_after']==a['state_before'] and iso['state_rng_modes_gradients_unchanged']
 for key in ('global_hash','transform_hashes','owner_raw_hashes'):assert d[key]==raw[key]
 assert d['incremental_communication_bytes']==d['incremental_persistent_storage_bytes']==0
 for r in d['diagnostics']['per_client']:
  assert r['shared_missing_correct']==r['refine_missing_correct']
  t=r['transitions'];assert sum(t.values())==r['seen_count']
  assert r['refine_seen_correct']-r['shared_seen_correct']==t['wrong_owned_to_correct']-t['correct_to_wrong']
 for name,arm in [('H07 aligned',a),('H09-A raw dual',raw),('H10-B group refine',d)]:
  assert all(math.isfinite(arm['metrics'][k]) for k in ('seen','missing','all','macro'))
  lines.append(f'| {seed} | {name} | '+' | '.join(f"{100*arm['metrics'][k]:.6f}" for k in ('seen','missing','all','macro'))+f" | {arm['predicted_class_count']} |")
 delta={k:100*(d['metrics'][k]-a['metrics'][k]) for k in ('seen','missing','all')};assert delta['missing']==0
 checks[str(seed)]=dict(h09a_h07_entire_references_exact=True,all_ten_online_exact=True,state_rng_modes_gradients_unchanged=True,missing_per_example_and_counts_exact=True,delta_pp=delta,strong=delta['seen']>=10 and delta['all']>=2 and d['predicted_class_count']>=9,diagnostics=d['diagnostics'])
 components=raw['component_diagnostics']['per_client'];ceiling=sum(r['native_owner_seen_only_correct'] for r in components)/sum(r['seen_count'] for r in components)
 details+=['',f'Seed {seed}: historical owner-only seen diagnostic ceiling {100*ceiling:.4f}%.', 'Routing and five transitions: '+json.dumps(d['diagnostics']['total']), 'Prediction histograms: '+json.dumps(d['prediction_histograms']['total']), 'Per-client exact missing (H07, H10-B): '+str([(r['shared_missing_correct'],r['refine_missing_correct']) for r in d['diagnostics']['per_client']])]
count=sum(v['strong'] for v in checks.values());verdict='strong' if count==3 else ('intermediate' if count else '0/3: stop routing/fusion line; retain H07')
checks.update(strong_seed_count=count,verdict=verdict)
lines+=['',f'Strong seeds: {count}/3. Frozen verdict: {verdict}.','', 'No labels enter prediction/routing; no calibration or extra local refresh. Incremental communication and persistent storage: 0 B relative to existing H07 objects. Existing anchor and affine-map delivery caveats remain. Full per-client/class metrics and counts are in final.json.']+details
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');(root/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print('\n'.join(lines))
