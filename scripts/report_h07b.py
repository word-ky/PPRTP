import json,sys
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


root=Path(sys.argv[1]);checks={};details=[]
lines=['# H07-B ordinary-local direct readout across seeds','',
    '| Seed | Arm | Seen % | Missing % | All % | Macro % | Predicted classes |',
    '|---|---|---:|---:|---:|---:|---:|']
zero=read('research_log/H07A/gate/artifacts/experiment/fedgh_seed0/final.json')['local_source_probe']
for seed in (0,1,2):
    if seed==0:
        arms={k:zero[k] for k in ('localtrain_aligned_global_prototype_cosine','localtrain_native_global_prototype_cosine_control')}
    else:
        folder=root/f'artifacts/experiment/fedgh_seed{seed}'
        historical=Path(f'research_log/H04B/full/artifacts/experiment/fedgh_seed{seed}')
        rr=[json.loads(s) for s in (folder/'rounds.jsonl').read_text().splitlines()]
        old=[json.loads(s) for s in Path(f'research_log/H02A/full/artifacts/experiment/fedgh_seed{seed}/rounds.jsonl').read_text().splitlines()]
        assert len(rr)==len(old)==10
        for r,h in zip(rr,old):
            assert all(r[k]==h[k] for k in ('metrics','client_model_hashes','prototype_bank_hash','server_head'))
        assert all('direct_cross_seed_probe' not in r for r in rr[:9])
        probe=rr[-1]['direct_cross_seed_probe'];arms=probe['arms']
        ref=read(historical/'final.json')['cross_seed_probe']['paired_256_2000']
        provenance=read(folder/'cross_seed_provenance.json')
        assert provenance==read(historical/'cross_seed_provenance.json')
        assert probe['anchor_receipt']==ref['anchor_receipt']
        source=probe['local_source'];split=read(folder/'split.json')
        assert source['train_indices']==split['train_indices'] and source['samples_per_client']==[200]*10
        assert source['class_sets']==split['class_sets']
        assert source['heldout_overlap']==source['anchor_overlap']==0
        assert source['official_train_test_separate'] and not source['heldout_semantics_used'] and not source['online_client_protos_used']
        p=arms[f'seed{seed}_localtrain_aligned_global_prototype_cosine']
        n=arms[f'seed{seed}_localtrain_native_global_prototype_cosine_control']
        assert p['alignment']==ref['alignment']
        assert [(r['client'],r['label'],r['count'],r['raw_hash']) for r in p['local_prototypes']]==[(r['client'],r['label'],r['count'],r['raw_hash']) for r in n['local_prototypes']]
        for arm in arms.values():
            assert arm['state_before']==arm['state_after']==ref['state_before']
            assert all(arm[k] for k in ('rng_cpu_unchanged','rng_cuda_unchanged','module_modes_unchanged','existing_gradients_unchanged'))
            assert len(arm['local_prototypes'])==20 and all(r['count']==100 for r in arm['local_prototypes'])
            assert arm['global_labels']==list(range(10)) and arm['prototype_norm_min']>0
            assert not arm['fitting'] and not arm['test_used_for_transform'] and arm['cosine_logits_finite']
        rm=p['metrics']['missing']/ref['metrics']['missing'];ra=p['metrics']['all']/ref['metrics']['all']
        gain=100*(p['metrics']['missing']-n['metrics']['missing'])
        checks[str(seed)]=dict(retM=rm,retA=ra,gain_pp=gain,predicted_classes=p['predicted_class_count'],
            strong=rm>=.70 and ra>=.75 and gain>=8 and p['predicted_class_count']>=8,
            fragile=rm<.40 or abs(gain)<=3,
            all_ten_online_exact=True,h04b_provenance_alignment_state_exact=True,
            raw_semantic_means_counts_matched=True,state_rng_modes_gradients_unchanged=True)
        details+=['',f'Seed {seed} provenance: '+json.dumps({k:v for k,v in provenance.items() if k.endswith('sha256')}),
            'N256 prefix SHA256: '+probe['anchor_receipt']['indices_sha256'],
            'Local train SHA256: '+source['indices_sha256'],
            'Per-client train SHA256: '+json.dumps(source['per_client_index_sha256']),
            'Class sets: '+json.dumps(source['class_sets']),
            'Transform SHA256: '+json.dumps([d['transform_hash'] for d in p['alignment']]),
            f"Refresh: {source['refresh_forward_examples_total']} forward examples per arm (200/client); local compute, not communication."]
    for name,arm in arms.items():
        fields=[f"{100*arm['metrics'][k]:.6f}" for k in ('seen','missing','all','macro')]
        lines.append(f'| {seed} | {name} | '+' | '.join(fields)+f" | {arm['predicted_class_count']} |")
        if seed:
            details+=['',name+':','Aggregate SHA256: '+arm['global_hash'],
                f"Norm range: {arm['prototype_norm_min']:.9g} to {arm['prototype_norm_max']:.9g}",
                'Prediction histograms (class0..9): '+json.dumps(arm['prediction_histograms']['total']),
                'Communication bytes: '+json.dumps(arm['communication']),
                '', '| Class | Owners | Counts | Total | Max/Fro hierarchical error | SHA256 |',
                '|---|---|---|---:|---|---|']
            for c in arm['global_prototypes']:
                details.append(f"| {c['label']} | {c['owners']} | {c['counts']} | {c['total_count']} | {c['hierarchical_max_error']:.9g}/{c['hierarchical_fro_error']:.9g} | {c['hash']} |")
values=list(checks.values())
verdict='A: strong cross-seed final-method replication' if all(v['strong'] for v in values) else ('B: final readout is seed-fragile' if any(v['fragile'] for v in values) else 'C: intermediate')
lines+=['','| Seed | retM | retA | Alignment gain pp | Predicted classes |','|---|---:|---:|---:|---:|']
for seed,v in checks.items():
    lines.append(f"| {seed} | {v['retM']:.9g} | {v['retA']:.9g} | {v['gain_pp']:.6f} | {v['predicted_classes']} |")
lines+=['',f'Frozen verdict: {verdict}.',
    'Seed0 is reused from H07-A; only seeds1/2 were run. Full per-client/per-class correct/count tables, local/global prototype receipts, and per-client histograms are in each final.json.',
    'Each aligned readout uses 41,280 B semantic uplink, 5,242,880 B anchor-feature uplink, and 204,800 B global-prototype downlink total. Anchor/reference distribution remains outside this feature accounting.']+details
checks['verdict']=verdict
(root/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps(checks,indent=2))
