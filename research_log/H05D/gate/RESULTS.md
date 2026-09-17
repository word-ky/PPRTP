# H05-D float64 solver-only audit

| Arm | Seen % | Missing % | All % | Macro % | Fit % | CE | grad_inf | grad_l2 | Weight/bias norm | Iter/eval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| rel255_paired_helmert_zscore_fp64 | 55.950000 | 11.837500 | 20.660000 | 20.660000 | 68.450004 | 1.08140367 | 0.00197867214 | 0.0260176584 | 751.47876/0.907168241 | 2000/2111 |
| rel255_broken_helmert_zscore_fp64 | 60.249999 | 4.762500 | 15.860000 | 15.859999 | 76.350003 | 0.854595055 | 0.0010495563 | 0.00791066368 | 812.502234/0.670096799 | 2000/2092 |

R64=11.837500%, S64=4.762500%, q_rel_64=0.538374088, delta=7.075000pp. Frozen branch: still solver-unresolved.

rel255_paired_helmert_zscore:
{
  "support_hash": "845fc12e427ccb65980b176fc70335c1fbd8fd248d8a2705750f9c2636af2e3a",
  "test_hash": "33d5529e6673c8ced16ce1622ad3c0ae711a26f006ddfc22c705fc470d1375ca",
  "support_labels_hash": "3c2c8976e417d87ef7900f1762cfa6ccd3102c21bf90340eb576be20a1f6cded",
  "test_labels_hash": "3ec878cc71c6e7af0269b9f28ac14993dfd998608f65ba75486f511e34292057",
  "dtype": "torch.float32"
}
{
  "input_dtype": "torch.float32",
  "solver_dtype": "torch.float64",
  "roundtrip_bitwise_equal": true,
  "support_hash": "0155bba740f810019a939fbe534da6d845c076fab9b1181670f08588fbe3a855",
  "test_hash": "156554db22421cc64b7a8ec2ed894df4c0620336608ab744fbff7464f631ad33"
}
Initial fit: {"ce": 2.302585092994046, "accuracy": 0.10000000149011612}

rel255_broken_helmert_zscore:
{
  "support_hash": "73cc6868693feea83cdea22a355a1473d8e08d439ac65f508b255c6939e2e602",
  "test_hash": "b287678634b74d8bc2635a87007eb1bb702a55698fe56d54b99ddec78e01ef0b",
  "support_labels_hash": "3c2c8976e417d87ef7900f1762cfa6ccd3102c21bf90340eb576be20a1f6cded",
  "test_labels_hash": "3ec878cc71c6e7af0269b9f28ac14993dfd998608f65ba75486f511e34292057",
  "dtype": "torch.float32"
}
{
  "input_dtype": "torch.float32",
  "solver_dtype": "torch.float64",
  "roundtrip_bitwise_equal": true,
  "support_hash": "66fdec23df3610e72342878a4f63a72f2371f7943f703dbc2eb9007a1694d429",
  "test_hash": "90197aebff6a0f9a8381a2e4e8aea4c4e52b3ad7557485a7b2f1f051a0f2909d"
}
Initial fit: {"ce": 2.302585092994046, "accuracy": 0.10000000149011612}

Both original H05-C float32 arms reproduced exactly, including all metrics, fit diagnostics, hashes and statistics, before the corresponding double fit. Final float32 feature matrices were hashed, then cast only; double-to-float32 roundtrips bitwise equal. All historical online/state/provenance checks passed. No optimizer/iteration/representation change.
