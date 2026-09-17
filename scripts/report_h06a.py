import json,sys
from pathlib import Path
root=Path(sys.argv[1]);folder=root/'artifacts/experiment/fedgh_seed0'
rr=[json.loads(s) for s in (folder/'rounds.jsonl').read_text().splitlines()]
old=[json.loads(s) for s in Path('research_log/H02A/full/artifacts/experiment/fedgh_seed0/rounds.jsonl').read_text().splitlines()]
assert len(rr)==10
for r,h in zip(rr,old):
    assert all(r[k]==h[k] for k in ('metrics','client_model_hashes','prototype_bank_hash','server_head'))
audit=rr[-1]['completion_probe'];arms=audit['arms']
historical=json.loads(Path('research_log/H04A/gate/artifacts/experiment/fedgh_seed0/final.json').read_text())['anchor_count_probe']['arms']['256']
assert audit['anchor_receipt']==historical['anchor_receipt']
can=arms['n256_completion_canonical'];assert can==historical['result']
assert json.loads((folder/'paired_anchors.json').read_text())==json.loads(Path('research_log/H04A/gate/artifacts/experiment/fedgh_seed0/paired_anchors.json').read_text())
pc=100*can['metrics']['missing'];values=[];eligible=[]
lines=['# H06-A nullspace completion audit','', '| Arm | Seen % | Missing % | All % | Macro % | Fit % | CE | grad_inf | grad_l2 | Weight/bias norm | Iter/eval | Retention |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---:|']
for name,a in arms.items():
    assert a['state_before']==a['state_after']==can['state_before']
    assert a['rank_diagnostics']==can['rank_diagnostics']
    assert all(a[k] for k in ('rng_cpu_unchanged','rng_cuda_unchanged','module_modes_unchanged','existing_gradients_unchanged'))
    f=a['fit'];g=f['final_support'];p=100*a['metrics']['missing']
    fields=[f"{100*a['metrics'][k]:.6f}" for k in ('seen','missing','all','macro')]
    fields += [f"{100*g['accuracy']:.6f}",f"{g['ce']:.9g}",f"{g['grad_inf']:.9g}",f"{g['grad_l2']:.9g}",f"{g['weight_norm']:.9g}/{g['bias_norm']:.9g}",f"{f['n_iter']}/{f['func_evals']}",f"{p/pc:.9g}"]
    lines.append('| '+name+' | '+' | '.join(fields)+' |')
    values.append(p)
    if name!='n256_completion_canonical' and g['accuracy']>=.95: eligible.append(p)
span=max(values)-min(values)
if len(eligible)==3 and min(eligible)/pc>=.8 and span<=5: verdict='A: completion-robust'
elif sum(pc-p>=8 for p in eligible)>=2 or (len(eligible)>=2 and max(eligible)-min(eligible)>=10): verdict='B: completion-sensitive'
else: verdict='C: intermediate; fit-limited arms excluded from sensitivity gate'
checks=dict(canonical_entire_h04a_exact=True,all_ten_online_exact=True,provenance_exact=True,state_rng_modes_gradients_unchanged=True,
    missing_values=values,adequately_fit_random_missing=eligible,range_all_pp=span,verdict=verdict)
lines+=['',f'Frozen verdict: {verdict}; range all arms={span:.6f}pp.','', '| Arm/client | Seed | Det sign | Orthogonality | Anchor max diff | Anchor Fro diff | Residual diff | Rank min / null max / tolerance |','|---|---:|---:|---:|---:|---:|---:|---|']
for name,a in arms.items():
    for c in a.get('completions',[]):
        if c['client']==0: continue
        assert c['rank']==255 and c['nullity']==257 and c['largest_null']<=c['tolerance']<c['smallest_constrained']
        assert c['translation_unchanged'] and not c['labels_used'] and not c['test_used']
        lines.append(f"| {name}/{c['client']} | {c['seed']} | {c['determinant_sign']} | {c['rotation_orthogonality_error']:.9g} | {c['mapped_anchor_max_difference']:.9g} | {c['mapped_anchor_fro_difference']:.9g} | {c['residual_difference']:.9g} | {c['smallest_constrained']:.9g}/{c['largest_null']:.9g}/{c['tolerance']:.9g} |")
lines+=['','Q: CPU local torch.Generator seed=602000+100*arm+client, iid double Gaussian QR with diagonal-R sign correction; independent nonreference clients; client0 identity. Q hashes/determinants/errors and all512 singular values saved perclient in final.json. Applied rotation remainsfloat32; equality checks above usefloat64 as preregistered.']
(root/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
