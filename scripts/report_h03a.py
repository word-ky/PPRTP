import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]); folder=root/'artifacts/experiment/fedgh_seed0'
rr=[json.loads(s) for s in (folder/'rounds.jsonl').read_text().splitlines()]
old=[json.loads(s) for s in Path('research_log/H02A/full/artifacts/experiment/fedgh_seed0/rounds.jsonl').read_text().splitlines()]
p=json.loads((folder/'paired_anchors.json').read_text()); split=json.loads((folder/'split.json').read_text())
oracle=json.loads(Path('research_log/H02C/full/artifacts/experiment/fedgh_seed0/oracle_calibration.json').read_text())
support=json.loads(Path('research_log/H02E/full/artifacts/experiment/fedgh_seed0/heldout_owner_support.json').read_text())
assert p['support_indices']==support['indices'] and p['support_indices_sha256']==support['indices_sha256']
assert not set(p['indices']).intersection(sum(split['train_indices'],[])+oracle['indices']+sum(support['indices'],[]))
assert len(set(p['indices']))==1000 and not p['anchor_labels_used']
assert p['indices_sha256']==hashlib.sha256(json.dumps(p['indices'],separators=(',',':')).encode()).hexdigest()
assert len(rr)==2
for r,h in zip(rr,old):
    assert all(r[k]==h[k] for k in ('metrics','client_model_hashes','prototype_bank_hash','server_head'))
o=rr[1]['paired_anchor_procrustes_probe'];f=o['fit']
assert o['state_before']==o['state_after'] and o['rng_cpu_unchanged'] and o['rng_cuda_unchanged'] and o['module_modes_unchanged']
q=(100*o['metrics']['missing']-.0875)/(31.45-.0875)
checks=dict(online_exact=True,state_rng_modes_unchanged=True,anchor_disjoint=True,support_exact=True,
    anchor_hash=p['indices_sha256'],q_align=q,head_fit_adequate=f['after']['accuracy']>=.95)
(root/'verification.json').write_text(json.dumps(checks,indent=2))
lines=['# H03-A paired-anchor orthogonal alignment','',f"Anchor hash `{p['indices_sha256']}`.1000 train-only images, seed161803,label-blind selection; no train/oracle/support overlap.",
       'Exact H02-E support reused; online H02-A round1/2 exact; state/modes/CPU-CUDA RNG unchanged.',
       '', '| Round | Seen % | Missing % | All % | Macro % |','|---|---:|---:|---:|---:|',
       '| 2 | '+' | '.join(f"{100*o['metrics'][k]:.6f}" for k in ('seen','missing','all','macro'))+' |',
       '',f"Head CE {f['before']['ce']:.9g} → {f['after']['ce']:.9g}; train accuracy {f['before']['accuracy']:.6f} → {f['after']['accuracy']:.6f}; LBFGS {f['n_iter']} iterations/{f['func_evals']} evaluations.",
       f'q_align={q:.9g}; head adequate={checks["head_fit_adequate"]}. Fixed missing references:.0875% owner-support,31.45% shared oracle.',
       '', '| Client | Raw residual before | Centered before | Centered after | Relative reduction | Orthogonality(double) | Orthogonality(applied float32) |',
       '|---|---:|---:|---:|---:|---:|---:|']
for i,d in enumerate(o['alignment']):
    lines.append(f'| {i} | '+' | '.join(f'{d[k]:.9g}' for k in ('raw_residual_before','centered_residual_before','centered_residual_after','relative_residual_reduction','orthogonality_error_double','orthogonality_error_applied'))+' |')
lines+=['','Residuals are Frobenius norms; relative reduction compares centered pre/post residuals. Client0 is identity (zero residual).',
        'Raw JSON retains per-client class counts/correct counts, transform hashes, head norms/hash, and side-effect receipts.']
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('\n'.join(lines))
