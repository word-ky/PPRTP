"""Frozen H12 checks/gates on preregistered ownership seed1, training seed0."""
import contextlib,io,json,runpy,sys
from pathlib import Path
root=Path(sys.argv[1]);sys.argv=['scripts/report_h12a.py',str(root),'0','1']
with contextlib.redirect_stdout(io.StringIO()):runpy.run_path(sys.argv[0],run_name='__main__')
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
base=root/'artifacts/experiment';old=read('research_log/H12A/full/artifacts/experiment/local_seed0/split.json');s=read(base/'local_seed0/split.json')
assert s['anchor_indices']==old['anchor_indices'] and s['anchor_indices_sha256']==old['anchor_indices_sha256']
assert s['test_indices']==old['test_indices'] and sorted(sum(s['train_indices'],[]))==sorted(sum(old['train_indices'],[]))
assert s['class_sets_sha256']!=old['class_sets_sha256'] and s['ownership_order_sha256']!=old['ownership_order_sha256']
assert all(len(cs)==len(set(cs))==20 for cs in s['class_sets'])
metas=[read(base/f'{m}_seed0/metadata.json') for m in ('local','fedproto','fedgh')]
assert all(m['ownership_seed']==1 and m['seed']==0 and not m['mixed_backbone'] for m in metas)
assert metas[0]['initial_state_sha256']==read('research_log/H12A/full/artifacts/experiment/local_seed0/metadata.json')['initial_state_sha256']
first=[json.loads((base/f'{m}_seed0/rounds.jsonl').read_text().splitlines()[0]) for m in ('local','fedproto','fedgh')]
assert first[0]['batch_hashes']==first[1]['batch_hashes']==first[2]['batch_hashes']
assert [len(h) for h in first[0]['batch_hashes']]==[(len(ii)+31)//32 for ii in s['train_indices']]
old_edges={(i,c) for i,cs in enumerate(old['class_sets']) for c in cs};new_edges={(i,c) for i,cs in enumerate(s['class_sets']) for c in cs}
jaccard=[len(set(a)&set(b))/len(set(a)|set(b)) for a,b in zip(old['class_sets'],s['class_sets'])]
distance=dict(old_incidence_count=len(old_edges),new_incidence_count=len(new_edges),removed=len(old_edges-new_edges),added=len(new_edges-old_edges),changed_binary_incidence_entries=len(old_edges^new_edges),changed_fraction_of_1000_entries=len(old_edges^new_edges)/1000,removed_fraction_of_200_old_edges=len(old_edges-new_edges)/200,classes_with_changed_owner_pair=sum(s['owners'][str(c)]!=old['owners'][str(c)] for c in range(100)),per_client_jaccard=jaccard,mean_client_jaccard=sum(jaccard)/10)
v=read(root/'verification.json');paired=read(base/'fedgh_seed0/final.json')['full_data_readout']['pprtp_h07']['metrics'];local=read(base/'local_seed0/final.json')['metrics']['head']
verdict='STRONG' if all(v['gates'].values()) else ('FAIL' if paired['missing']<.03 or v['missing_gap_pp']<1 else ('MIXED' if v['missing_gap_pp']>=3 else 'NOT STRONG; between preregistered MIXED and FAIL conditions'))
v.update(verdict=verdict,ownership_seed=1,training_seed=0,graph_distance=distance,anchors_exact_h12=True,nonanchor_pool_exact_h12=True,initial_model_exact_h12=True,actual_round1_batch_pairing=True,paired_minus_local_all_pp=100*(paired['all']-local['all']))
lines=['','## Ownership replication receipts','',f'H14-A verdict: {verdict}. All frozen metric definitions and gates unchanged.',f"Paired all minus Local all: {v['paired_minus_local_all_pp']:.6f} pp (diagnostic, not replacement gate).",'Ownership RNG seed1; training seed0; first balanced cyclic construction, no graph search. Labeled class-client graph changes; unlabeled client-neighbor cycle topology is retained.','Ownership distance: '+json.dumps(distance),'Per-class owner pairs: '+json.dumps(s['owners'],sort_keys=True),'','Exact historical anchors/non-anchor pool/test indices; historical seed0 initialmodel hash retained. Actual round1 batches pair acrossall3arms. This is one new graph and one trainingseed, not general graph/topology robustness. No tuning or alternate reference.']
text=(root/'RESULTS.md').read_text().replace('# H12-A CIFAR100 seed0 portability stress test','# H14-A CIFAR100 ownership seed1 / training seed0 replication',1)+'\n'.join(lines)+'\n'
(root/'RESULTS.md').write_text(text,encoding='utf-8');(root/'verification.json').write_text(json.dumps(v,indent=2),encoding='utf-8')
print('\n'.join(text.splitlines()[:16]));print(json.dumps(distance));print(verdict)
