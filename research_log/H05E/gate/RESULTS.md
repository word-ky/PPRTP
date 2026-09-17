# H05-E invertible SVD preconditioner audit

| Arm | Seen % | Missing % | All % | Macro % | Fit % | CE | grad_inf | grad_l2 | Weight/bias norm | Iter/eval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| rel255_paired_svdprecond_fp64 | 41.750000 | 12.337500 | 18.220000 | 18.220000 | 100.000000 | 8.10998518e-09 | 9.22523789e-10 | 8.54540119e-09 | 93258.3647/22469.0399 | 219/224 |
| rel255_broken_svdprecond_fp64 | 51.700000 | 6.512500 | 15.550000 | 15.549999 | 83.500004 | 0.617801318 | 5.84546883e-08 | 5.05746835e-07 | 17.3831045/4.05227261 | 218/299 |

Rpre=12.337500%, Spre=6.512500%, q_rel_pre=0.561114266, delta=5.825000pp. Frozen branch: adequately fit, intermediate.

rel255_paired_svdprecond:
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
Preconditioner: {
  "min_s": 0.010774107653777782,
  "max_s": 617.1836651829133,
  "condition_before": 57283.97051671439,
  "condition_after": 1.0000000000380773,
  "max_scaling": 4150.817959788523,
  "whitening_residual_fro": 5.673416567233893e-11,
  "labels_used": false,
  "test_used": false,
  "dimensions": 255,
  "transform_hash": "4d7dfb24546ef480d341ca348b883f5bdc73025f3c0cfd8e31d2dbf4ca8024dc",
  "inverse_hash": "d08112efd13283bca9b645e5f562c2cb5f990e2b405774faf06eb4024abc9f45",
  "support_reconstruction_relative": [
    1.4262924173459914e-13,
    1.4262860031009708e-13,
    1.4282612945297003e-13,
    1.428108832852271e-13,
    1.425935518420351e-13,
    1.4243466962843988e-13,
    1.4038143755312523e-13,
    1.3975516862270515e-13,
    1.4076358701524164e-13,
    1.4297279719593446e-13
  ],
  "test_reconstruction_relative": [
    1.4263559228027894e-13,
    1.4260891645907954e-13,
    1.4279718618008505e-13,
    1.428037581985409e-13,
    1.4267472314536387e-13,
    1.4244389140551607e-13,
    1.4039594939183827e-13,
    1.3970589278406973e-13,
    1.4073253775457844e-13,
    1.4291701825463928e-13
  ]
}

rel255_broken_svdprecond:
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
Preconditioner: {
  "min_s": 0.047553200821000445,
  "max_s": 325.768850811085,
  "condition_before": 6850.618784576514,
  "condition_after": 1.0000000000061788,
  "max_scaling": 940.4489871951154,
  "whitening_residual_fro": 1.7591720936671875e-11,
  "labels_used": false,
  "test_used": false,
  "dimensions": 255,
  "transform_hash": "cf6426b16e74e61d6163d8d0807e825799da464b9cd5a43fdc5163d0ecb960e2",
  "inverse_hash": "c26d8855c79d06beac7e74bcebcdd1c79eb2a19575fcd219d4737a5178628a27",
  "support_reconstruction_relative": [
    1.3482743428940763e-13,
    1.3812703416128298e-13,
    1.288137860675646e-13,
    1.3454738291574349e-13,
    1.3121075910779997e-13,
    1.3194437902211395e-13,
    1.3485199491611824e-13,
    9.340273954759508e-14,
    1.2933833487855443e-13,
    1.4656753263940475e-13
  ],
  "test_reconstruction_relative": [
    1.348203914333658e-13,
    1.381809315358816e-13,
    1.2886482242600415e-13,
    1.344541746556444e-13,
    1.3117672788454056e-13,
    1.3196916259883108e-13,
    1.3484696475637088e-13,
    9.359759273460381e-14,
    1.2934996700284726e-13,
    1.4648773822934777e-13
  ]
}

Exact H05-D float32 and float64 support/test/label hashes, statistics, basis, Grams, permutations and frozen state verified. All255 positive singular values retained; no recentering, thresholds, clamps or regularization. Full pre/post singular spectra and perclient support/test reconstruction receipts saved in final.json. All historical online/state/provenance checks passed. Same affine-linear capacity and optimizer/iteration budget.
