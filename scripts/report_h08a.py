import json,sys,math
from pathlib import Path
root=Path(sys.argv[1]);arms={};checks={}
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
ref=read('research_log/H07A/gate/artifacts/experiment/fedgh_seed0/final.json')['local_source_probe']['localtrain_aligned_global_prototype_cosine']
lines=['# H08-A lag-1 aligned-GPC causal gate','','| Arm | Round | Seen % | Missing % | All % | Macro % | Predicted classes |','|---|---:|---:|---:|---:|---:|---:|']
lines.append('| fedgh_posthoc_reference | 10 | '+' | '.join(f"{100*ref['metrics'][k]:.6f}" for k in ('seen','missing','all','macro'))+' | 10 |')
details=[]
for name in ('pprtp_all_lag1','pprtp_seen_lag1'):
    folder=root/f'artifacts/experiment/{name}_seed0'
    rr=[json.loads(s) for s in (folder/'rounds.jsonl').read_text().splitlines()];arms[name]=rr
    assert len(rr)==10 and rr[0]['historical_round_one_exact']
    provenance=read(folder/'online_provenance.json')
    assert provenance['local_source']['anchor_overlap']==0
    for i,r in enumerate(rr):
        bank=r['aligned_bank']
        assert bank['state_before']==bank['state_after'] and bank['state_rng_modes_gradients_unchanged'] and bank['detached']
        assert bank['anchor_indices_sha256']==provenance['anchor_receipt']['indices_sha256']
        assert len(bank['local_prototypes'])==20 and len(bank['global_prototypes'])==10
        assert all(p['count']==100 for p in bank['local_prototypes'])
        assert all(p['count']==200 and math.isfinite(p['norm']) and p['norm']>0 for p in bank['global_prototypes'])
        assert all(math.isfinite(v) for loss in r['losses'] for v in loss.values())
        if i:
            assert r['training_bank']['global_hash']==rr[i-1]['aligned_bank']['global_hash']
            assert r['training_bank']['transform_hashes']==[d['transform_hash'] for d in rr[i-1]['aligned_bank']['alignment']]
            assert r['training_bank']['frozen_through_epoch'] and r['training_bank']['source_round']==i
        if r['round'] in (2,5,10):
            a=r['aligned_direct']
            lines.append(f"| {name} | {r['round']} | "+' | '.join(f"{100*a['metrics'][k]:.6f}" for k in ('seen','missing','all','macro'))+f" | {a['predicted_class_count']} |")
            details+=['',f"{name}, round {r['round']}",'Ordinary readouts: '+json.dumps(r['metrics']),
                'Mean losses: '+json.dumps({k:sum(v[k] for v in r['losses'])/10 for k in ('local','knowledge')}),
                'Client0 first-batch diagnostics: '+json.dumps(r['diagnostic_client0']),
                'Prediction histograms (class0..9): '+json.dumps(a['prediction_histograms']['total']),
                'Fresh bank SHA256: '+bank['global_hash'],
                'Owner norm range: '+str([min(p['norm'] for p in bank['local_prototypes']),max(p['norm'] for p in bank['local_prototypes'])]),
                'Global norm range: '+str([min(p['norm'] for p in bank['global_prototypes']),max(p['norm'] for p in bank['global_prototypes'])])]
    checks[name]=dict(round1_exact=True,lag1_hashes_exact=True,state_rng_modes_gradients_unchanged=True,finite=True)
a=arms['pprtp_all_lag1'];s=arms['pprtp_seen_lag1']
assert a[0]['aligned_bank']==s[0]['aligned_bank']
assert all(x['batch_hashes']==y['batch_hashes'] for x,y in zip(a,s))
assert a[1]['diagnostic_client0']['preupdate_feature_hash']==s[1]['diagnostic_client0']['preupdate_feature_hash']
assert a[1]['diagnostic_client0']['denominator_feature_gradients']==s[1]['diagnostic_client0']['denominator_feature_gradients']
x=a[-1]['aligned_direct'];y=s[-1]['aligned_direct'];m=x['metrics'];n=y['metrics']
dm=100*(m['missing']-n['missing']);da=100*(m['all']-n['all'])
d=a[1]['diagnostic_client0'];nonzero=d['feature_extractor_knowledge_grad_norm']>0 and d['missing_probability_on_seen']>0
if 100*m['missing']>=18 and 100*m['all']>=23.39 and dm>=2 and da>=1 and x['predicted_class_count']==10 and nonzero: verdict='A: strong online aligned-GPC signal'
elif 100*m['missing']<10 or 100*m['all']<20 or (dm<=.5 and da<=.5 and nonzero): verdict='B: online mechanism falsified at frozen strength'
else: verdict='C: intermediate'
checks.update(first_bank_identical=True,all_batches_identical=True,round2_same_preupdate_tensor=True,delta_missing_pp=dm,delta_all_pp=da,nonzero_allclass_signal=nonzero,verdict=verdict)
b=a[-1]['aligned_bank']
lines+=['',f'Frozen verdict: {verdict}; all-minus-seen missing={dm:.6f}pp, all={da:.6f}pp.',
    'Per bank: '+json.dumps({k:b[k] for k in ('semantic_forward_examples','anchor_forward_examples','semantic_uplink_bytes','anchor_uplink_bytes','bank_bytes','bank_downlink_total','naive_transform_bytes_per_client')}),
    'Ten builds per arm: nine training banks and one evaluation-only final bank; rounds2/5 reuse the fresh next-round bank for separate evaluation. Each arm refreshes20,000 local semantic and25,600 anchor forward examples. Naive affine transform includes both512-D means and512x512 matrix;10,526,720B total per build. This excludes image/reference distribution and unchanged FedGH traffic; no communication-efficiency claim.',
    'All per-client/class counts, histograms, bank/transform hashes, losses and ordinary readouts are preserved in rounds.jsonl/final.json. Seen-arm missing_probability_on_seen records the counterfactual ALL-class softmax on the same tensor, not the masked training softmax (whose missing mass is zero).']+details
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(root/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print(json.dumps(checks,indent=2))
