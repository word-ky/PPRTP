import json,sys
from pathlib import Path
root=Path(sys.argv[1]);folder=root/'artifacts/experiment/fedgh_seed0'
rr=[json.loads(s) for s in (folder/'rounds.jsonl').read_text().splitlines()]
old=[json.loads(s) for s in Path('research_log/H02A/full/artifacts/experiment/fedgh_seed0/rounds.jsonl').read_text().splitlines()]
assert len(rr)==10
for r,h in zip(rr,old): assert all(r[k]==h[k] for k in ('metrics','client_model_hashes','prototype_bank_hash','server_head'))
a=rr[-1]['local_source_probe'];h=json.loads(Path('research_log/H06C/gate/artifacts/experiment/fedgh_seed0/final.json').read_text())['direct_prototype_probe']
ref=a['heldout_aligned_global_prototype_cosine_reference'];assert ref==h['aligned_global_prototype_cosine'] and a['anchor_receipt']==h['anchor_receipt']
assert json.loads((folder/'paired_anchors.json').read_text())==json.loads(Path('research_log/H06C/gate/artifacts/experiment/fedgh_seed0/paired_anchors.json').read_text())
lines=['# H07-A ordinary local semantic source','', '| Arm | Seen % | Missing % | All % | Macro % | Predicted classes |','|---|---:|---:|---:|---:|---:|']
for name in ('heldout_aligned_global_prototype_cosine_reference','localtrain_aligned_global_prototype_cosine','localtrain_native_global_prototype_cosine_control'):
    arm=a[name];assert arm['state_before']==arm['state_after']==ref['state_before']
    assert all(arm[k] for k in ('rng_cpu_unchanged','rng_cuda_unchanged','module_modes_unchanged','existing_gradients_unchanged'))
    fields=[f"{100*arm['metrics'][k]:.6f}" for k in ('seen','missing','all','macro')]
    lines.append('| '+name+' | '+' | '.join(fields)+' | '+str(arm.get('predicted_class_count','not collected'))+' |')
p=a['localtrain_aligned_global_prototype_cosine'];n=a['localtrain_native_global_prototype_cosine_control']
rm=p['metrics']['missing']/ref['metrics']['missing'];ra=p['metrics']['all']/ref['metrics']['all'];gain=100*(p['metrics']['missing']-n['metrics']['missing'])
if rm>=.75 and ra>=.8 and gain>=8 and p['predicted_class_count']>=8: verdict='A: strong ordinary-local semantic closure'
elif rm<.5 or ra<.6: verdict='B: fresh held-out semantics materially important'
else: verdict='C: intermediate'
checks=dict(h06c_entire_reference_exact=True,all_ten_online_exact=True,provenance_exact=True,state_rng_modes_gradients_unchanged=True,
    ret_missing=rm,ret_all=ra,alignment_gain_pp=gain,predicted_classes=p['predicted_class_count'],verdict=verdict)
lines+=['',f'Frozen verdict: {verdict}. ret_missing={rm:.9g}; ret_all={ra:.9g}; alignment_gain={gain:.6f}pp.']
for name in ('localtrain_aligned_global_prototype_cosine','localtrain_native_global_prototype_cosine_control'):
    arm=a[name];assert arm['global_labels']==list(range(10)) and len(arm['global_prototypes'])==10
    assert not arm['fitting'] and not arm['test_used_for_transform'] and arm['cosine_logits_finite']
    lines+=['',name+':', 'Aggregate SHA256: '+arm['global_hash'],f"Norm range: {arm['prototype_norm_min']:.9g} to {arm['prototype_norm_max']:.9g}",
        'Prediction histograms (class0..9): '+json.dumps(arm['prediction_histograms']['total']),
        'Communication bytes: '+json.dumps(arm['communication']),
        '', '| Class | Owners | Counts | Total count | Hierarchical max error | Fro error | Prototype SHA256 |','|---|---|---|---:|---:|---:|---|']
    for c in arm['global_prototypes']:
        assert c['label']==c['bank_row']
        lines.append(f"| {c['label']} | {c['owners']} | {c['counts']} | {c['total_count']} | {c['hierarchical_max_error']:.9g} | {c['hierarchical_fro_error']:.9g} | {c['hash']} |")
source=a['local_source'];split=json.loads((folder/'split.json').read_text())
assert source['train_indices']==split['train_indices'] and source['samples_per_client']==[200]*10
assert source['heldout_overlap']==source['anchor_overlap']==0
assert source['official_train_test_separate'] and not source['heldout_semantics_used'] and not source['online_client_protos_used']
assert p['alignment']==ref['alignment']
assert [(r['client'],r['label'],r['count'],r['raw_hash']) for r in p['local_prototypes']]==[(r['client'],r['label'],r['count'],r['raw_hash']) for r in n['local_prototypes']]
checks['local_source_provenance_exact']=True
lines+=['','Ordinary local train index SHA256: '+source['indices_sha256'],
    'Perclient index SHA256: '+json.dumps(source['per_client_index_sha256']),
    'Prototype refresh forward examples: '+str(source['refresh_forward_examples_total'])+' total; '+str(source['refresh_forward_examples_per_client'])+' perclient. Local compute, not communication. Each diagnostic arm recomputes this same refresh.',
    '', '| Class | New-vs-heldout cosine | L2 distance |','|---|---:|---:|']
for d in a['source_shift']: lines.append(f"| {d['label']} | {d['cosine']:.9g} | {d['l2']:.9g} |")
(root/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
