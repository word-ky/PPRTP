import json,sys
from pathlib import Path
root=Path(sys.argv[1]);folder=root/'artifacts/experiment/fedgh_seed0'
rr=[json.loads(s) for s in (folder/'rounds.jsonl').read_text().splitlines()]
old=[json.loads(s) for s in Path('research_log/H02A/full/artifacts/experiment/fedgh_seed0/rounds.jsonl').read_text().splitlines()]
assert len(rr)==10
for r,h in zip(rr,old):
    assert all(r[k]==h[k] for k in ('metrics','client_model_hashes','prototype_bank_hash','server_head'))
a=rr[-1]['class_prototype_probe']
h=json.loads(Path('research_log/H04A/gate/artifacts/experiment/fedgh_seed0/final.json').read_text())['anchor_count_probe']['arms']['256']
ref=a['full_aligned_support_reference'];assert ref==h['result'] and a['anchor_receipt']==h['anchor_receipt']
assert json.loads((folder/'paired_anchors.json').read_text())==json.loads(Path('research_log/H04A/gate/artifacts/experiment/fedgh_seed0/paired_anchors.json').read_text())
proto=a['aligned_client_class_prototypes'];native=a['native_client_class_prototypes_control']
assert [(p['client'],p['label'],p['count'],p['raw_hash']) for p in proto['prototypes']]==[(p['client'],p['label'],p['count'],p['raw_hash']) for p in native['prototypes']]
assert proto['alignment']==ref['alignment']
lines=['# H06-B owner-class prototype compression','', '| Arm | Seen % | Missing % | All % | Macro % | Training fit % | Training CE | Full support fit % | grad_inf | grad_l2 | W/b norm | Iter/eval |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|']
for name in ('full_aligned_support_reference','aligned_client_class_prototypes','native_client_class_prototypes_control'):
    arm=a[name];f=arm['fit']
    assert arm['state_before']==arm['state_after']==ref['state_before']
    assert all(arm[k] for k in ('rng_cpu_unchanged','rng_cuda_unchanged','module_modes_unchanged','existing_gradients_unchanged'))
    g=f['final_support'] if name=='full_aligned_support_reference' else f
    full=f['after']['accuracy'] if name=='full_aligned_support_reference' else arm['full_support_diagnostic']['accuracy']
    fields=[f"{100*arm['metrics'][k]:.6f}" for k in ('seen','missing','all','macro')]
    fields += [f"{100*f['after']['accuracy']:.6f}",f"{f['after']['ce']:.9g}",f"{100*full:.6f}",f"{g['grad_inf']:.9g}",f"{g['grad_l2']:.9g}",f"{f['weight_norm']:.9g}/{f['bias_norm']:.9g}",f"{f['n_iter']}/{f['func_evals']}"]
    lines.append('| '+name+' | '+' | '.join(fields)+' |')
rm=proto['metrics']['missing']/ref['metrics']['missing'];ra=proto['metrics']['all']/ref['metrics']['all'];gain=100*(proto['metrics']['missing']-native['metrics']['missing'])
if min(proto['fit']['after']['accuracy'],native['fit']['after']['accuracy'])<.95: verdict='fit-limited; stop'
elif rm>=.8 and ra>=.8 and gain>=8: verdict='A: strong class-prototype compression'
elif rm<.5: verdict='B: class means are too lossy'
else: verdict='C: intermediate'
checks=dict(canonical_entire_h04a_exact=True,all_ten_online_exact=True,provenance_exact=True,raw_prototypes_labels_counts_matched=True,
    state_rng_modes_gradients_unchanged=True,ret_missing=rm,ret_all=ra,alignment_gain_pp=gain,verdict=verdict)
lines+=['',f'Frozen verdict: {verdict}. ret_missing={rm:.9g}; ret_all={ra:.9g}; alignment_gain={gain:.6f}pp.',
    '', 'Semantic vectors, labels and counts use actual float32/int64/int64 dtypes. Anchor feature payload is added to BOTH aligned full-support and aligned prototype totals; native control uses no anchors. This accounting concerns upload tensors, not network framing, raw-image distribution or a complete online protocol.',
    '', 'Aligned payload perclient and totals:',json.dumps(proto['communication'],indent=2),
    '', '| Client/class | Count | Affine mean max error | Affine mean Fro error |','|---|---:|---:|---:|']
for p in proto['prototypes']:
    lines.append(f"| {p['client']}/{p['label']} | {p['count']} | {p['affine_mean_max_error']:.9g} | {p['affine_mean_fro_error']:.9g} |")
for name in ('aligned_client_class_prototypes','native_client_class_prototypes_control'):
    arm=a[name]
    lines+=['',name+': '+json.dumps({k:arm[k] for k in ('prototype_count','prototype_hash','labels_hash','counts_hash','prototype_dtype','label_dtype','count_dtype')})]
(root/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines[:11]));print(json.dumps(proto['communication']['totals'],indent=2))
