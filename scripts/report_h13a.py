"""Reuse H12 frozen metrics/gates; add H13 mixed-backbone receipts and grouping."""
import contextlib,io,json,runpy,sys
from pathlib import Path
root=Path(sys.argv[1]);base=root/'artifacts/experiment'
with contextlib.redirect_stdout(io.StringIO()):runpy.run_path('scripts/report_h12a.py',run_name='__main__')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
modes=('local','fedproto','fedgh');meta={m:read(base/f'{m}_seed0/metadata.json') for m in modes};runs={m:read(base/f'{m}_seed0/final.json') for m in modes}
split=read(base/'local_seed0/split.json');assert split==read('research_log/H12A/full/artifacts/experiment/local_seed0/split.json')
assignment=['FedAvgCNN','ResNet18']*5;receipts=meta['local']['client_initial_states']
for m in modes:
 assert meta[m]['architecture_assignment']==assignment and meta[m]['client_initial_states']==receipts
 assert meta[m]['mixed_backbone'] and meta[m]['server_head_initial_client']==0
for i,r in enumerate(receipts):
 assert r['client']==i and r['architecture']==assignment[i] and r['feature_dim']==512 and not r['pretrained']
 assert r['head_shapes']=={'weight':[100,512],'bias':[100]}
 assert any('running_mean' in k for k in r['buffer_names'])==(i%2==1)
for owners in split['owners'].values():assert len(owners)==2 and {assignment[i] for i in owners}=={'FedAvgCNN','ResNet18'}
first=[json.loads((base/f'{m}_seed0/rounds.jsonl').read_text().splitlines()[0]) for m in modes]
assert first[0]['batch_hashes']==first[1]['batch_hashes']==first[2]['batch_hashes']
assert [len(h) for h in first[0]['batch_hashes']]==[156]*10
f=runs['fedgh']['full_data_readout'];p=f['pprtp_h07'];b=runs['fedgh']['full_pair_probe']['pair_broken_h07'];n=f['native_global_prototype_cosine_control']
perclient={m:[r[key] for r in runs[m]['per_client']] for m,key in [('local','head'),('fedproto','l2'),('fedgh','global_head_post_server')]}
perclient.update(paired_h07=p['per_client'],pair_broken_h07=b['per_client'],native_control=n['per_client'])
extra=['','## Architecture initialization and grouped results','','Models constructed before upstream client initialization; no pretrained weights. All state hashes include BatchNorm buffers. Client0 FedAvgCNN is the unchanged reference; server head starts from client0 initialhead. All heads512->100. Same clientinitialhashes and actualround1batchhashes acrossarms.','','| Client | Architecture | Parameters | Initial model SHA256 | Initial base SHA256 | Initial head SHA256 |','|---|---|---:|---|---|---|']
for r in receipts:extra.append(f"| {r['client']} | {r['architecture']} | {r['parameter_count']} | {r['initial_model_hash']} | {r['initial_base_hash']} | {r['initial_head_hash']} |")
extra+=['','| Arm | Backbone | Seen % | Missing % | All % | Macro % |','|---|---|---:|---:|---:|---:|'];grouped={}
for name,values in perclient.items():
 grouped[name]={}
 for arch in ('FedAvgCNN','ResNet18'):
  ii=[i for i,a in enumerate(assignment) if a==arch];mm={k:sum(values[i][k] for i in ii)/len(ii) for k in ('seen','missing','all','macro')};grouped[name][arch]=mm
  extra.append(f'| {name} | {arch} | '+' | '.join(f'{100*mm[k]:.6f}' for k in ('seen','missing','all','macro'))+' |')
extra+=['','| Arm | Client | Backbone | Seen % | Missing % | All % | Macro % |','|---|---:|---|---:|---:|---:|---:|']
for name,values in perclient.items():
 for i,m in enumerate(values):extra.append(f'| {name} | {i} | {assignment[i]} | '+' | '.join(f'{100*m[k]:.6f}' for k in ('seen','missing','all','macro'))+' |')
residuals={};extra+=['','Mean centered alignment residual after transform, grouped over non-reference clients (client0 identity excluded):']
for arch in ('FedAvgCNN','ResNet18'):
 ii=[i for i,a in enumerate(assignment) if a==arch and i!=0]
 residuals[arch]={name:sum(r['alignment'][i]['centered_residual_after'] for i in ii)/len(ii) for name,r in [('paired',p),('broken',b)]}
 extra.append(arch+': '+json.dumps(residuals[arch]))
extra+=['','Residual magnitudes depend on feature scale; this grouping is descriptive, not proof of a failure mechanism. Full per-client residuals/orthogonality, classwise counts and histograms remain in final.json. The frozen overallgate is unchanged; groupedmetrics are not selected as alternativegates. Same512Dpayloads asH12; modelparametercounts differ, with no modelweight exchange in these prototype/head baselines. One mixedseed only; do not claim multi-seed architectureheterogeneity yet.']
text=(root/'RESULTS.md').read_text().replace('# H12-A CIFAR100 seed0 portability stress test','# H13-A CIFAR100 seed0 mixed-backbone falsifier',1)
text+='\n'.join(extra)+'\n';(root/'RESULTS.md').write_text(text,encoding='utf-8')
v=read(root/'verification.json');v.update(architecture_assignment=assignment,client_initial_states=receipts,initial_states_paired_by_client=True,round1_actual_batches_exact=True,split_exact_h12a=True,every_class_cross_architecture=True,per_backbone_metrics=grouped,nonreference_alignment_residuals=residuals)
(root/'verification.json').write_text(json.dumps(v,indent=2),encoding='utf-8')
print('\n'.join(text.splitlines()[:16]));print(json.dumps(grouped,indent=2))
