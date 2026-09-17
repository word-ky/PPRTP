# H06-B owner-class prototype compression

| Arm | Seen % | Missing % | All % | Macro % | Training fit % | Training CE | Full support fit % | grad_inf | grad_l2 | W/b norm | Iter/eval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| full_aligned_support_reference | 37.550000 | 21.987500 | 25.100000 | 25.100001 | 100.000000 | 7.68896147e-09 | 100.000000 | 1.4978065e-08 | 9.88137003e-08 | 1050686/24634.9922 | 1227/1308 |
| aligned_client_class_prototypes | 27.950000 | 23.887500 | 24.700000 | 24.700000 | 100.000000 | 4.76837094e-08 | 29.150000 | 2.6978098e-08 | 1.90271805e-07 | 1080.33606/131.921036 | 38/52 |
| native_client_class_prototypes_control | 67.200000 | 0.000000 | 13.440000 | 13.440000 | 100.000000 | 5.18559034e-07 | 67.350006 | 3.47188291e-07 | 2.58797149e-06 | 1764.81299/79.758255 | 34/54 |

Frozen verdict: A: strong class-prototype compression. ret_missing=1.08641273; ret_all=0.984063742; alignment_gain=23.887500pp.

Semantic vectors, labels and counts use actual float32/int64/int64 dtypes. Anchor feature payload is added to BOTH aligned full-support and aligned prototype totals; native control uses no anchors. This accounting concerns upload tensors, not network framing, raw-image distribution or a complete online protocol.

Aligned payload perclient and totals:
{
  "per_client": [
    {
      "client": 0,
      "support_vectors": 200,
      "prototype_vectors": 2,
      "full_vector_bytes": 409600,
      "full_label_bytes": 1600,
      "prototype_vector_bytes": 4096,
      "prototype_label_bytes": 16,
      "prototype_count_bytes": 16,
      "anchor_bytes": 524288,
      "full_semantic_bytes": 411200,
      "prototype_semantic_bytes": 4128,
      "full_with_anchors_bytes": 935488,
      "prototype_with_anchors_bytes": 528416
    },
    {
      "client": 1,
      "support_vectors": 200,
      "prototype_vectors": 2,
      "full_vector_bytes": 409600,
      "full_label_bytes": 1600,
      "prototype_vector_bytes": 4096,
      "prototype_label_bytes": 16,
      "prototype_count_bytes": 16,
      "anchor_bytes": 524288,
      "full_semantic_bytes": 411200,
      "prototype_semantic_bytes": 4128,
      "full_with_anchors_bytes": 935488,
      "prototype_with_anchors_bytes": 528416
    },
    {
      "client": 2,
      "support_vectors": 200,
      "prototype_vectors": 2,
      "full_vector_bytes": 409600,
      "full_label_bytes": 1600,
      "prototype_vector_bytes": 4096,
      "prototype_label_bytes": 16,
      "prototype_count_bytes": 16,
      "anchor_bytes": 524288,
      "full_semantic_bytes": 411200,
      "prototype_semantic_bytes": 4128,
      "full_with_anchors_bytes": 935488,
      "prototype_with_anchors_bytes": 528416
    },
    {
      "client": 3,
      "support_vectors": 200,
      "prototype_vectors": 2,
      "full_vector_bytes": 409600,
      "full_label_bytes": 1600,
      "prototype_vector_bytes": 4096,
      "prototype_label_bytes": 16,
      "prototype_count_bytes": 16,
      "anchor_bytes": 524288,
      "full_semantic_bytes": 411200,
      "prototype_semantic_bytes": 4128,
      "full_with_anchors_bytes": 935488,
      "prototype_with_anchors_bytes": 528416
    },
    {
      "client": 4,
      "support_vectors": 200,
      "prototype_vectors": 2,
      "full_vector_bytes": 409600,
      "full_label_bytes": 1600,
      "prototype_vector_bytes": 4096,
      "prototype_label_bytes": 16,
      "prototype_count_bytes": 16,
      "anchor_bytes": 524288,
      "full_semantic_bytes": 411200,
      "prototype_semantic_bytes": 4128,
      "full_with_anchors_bytes": 935488,
      "prototype_with_anchors_bytes": 528416
    },
    {
      "client": 5,
      "support_vectors": 200,
      "prototype_vectors": 2,
      "full_vector_bytes": 409600,
      "full_label_bytes": 1600,
      "prototype_vector_bytes": 4096,
      "prototype_label_bytes": 16,
      "prototype_count_bytes": 16,
      "anchor_bytes": 524288,
      "full_semantic_bytes": 411200,
      "prototype_semantic_bytes": 4128,
      "full_with_anchors_bytes": 935488,
      "prototype_with_anchors_bytes": 528416
    },
    {
      "client": 6,
      "support_vectors": 200,
      "prototype_vectors": 2,
      "full_vector_bytes": 409600,
      "full_label_bytes": 1600,
      "prototype_vector_bytes": 4096,
      "prototype_label_bytes": 16,
      "prototype_count_bytes": 16,
      "anchor_bytes": 524288,
      "full_semantic_bytes": 411200,
      "prototype_semantic_bytes": 4128,
      "full_with_anchors_bytes": 935488,
      "prototype_with_anchors_bytes": 528416
    },
    {
      "client": 7,
      "support_vectors": 200,
      "prototype_vectors": 2,
      "full_vector_bytes": 409600,
      "full_label_bytes": 1600,
      "prototype_vector_bytes": 4096,
      "prototype_label_bytes": 16,
      "prototype_count_bytes": 16,
      "anchor_bytes": 524288,
      "full_semantic_bytes": 411200,
      "prototype_semantic_bytes": 4128,
      "full_with_anchors_bytes": 935488,
      "prototype_with_anchors_bytes": 528416
    },
    {
      "client": 8,
      "support_vectors": 200,
      "prototype_vectors": 2,
      "full_vector_bytes": 409600,
      "full_label_bytes": 1600,
      "prototype_vector_bytes": 4096,
      "prototype_label_bytes": 16,
      "prototype_count_bytes": 16,
      "anchor_bytes": 524288,
      "full_semantic_bytes": 411200,
      "prototype_semantic_bytes": 4128,
      "full_with_anchors_bytes": 935488,
      "prototype_with_anchors_bytes": 528416
    },
    {
      "client": 9,
      "support_vectors": 200,
      "prototype_vectors": 2,
      "full_vector_bytes": 409600,
      "full_label_bytes": 1600,
      "prototype_vector_bytes": 4096,
      "prototype_label_bytes": 16,
      "prototype_count_bytes": 16,
      "anchor_bytes": 524288,
      "full_semantic_bytes": 411200,
      "prototype_semantic_bytes": 4128,
      "full_with_anchors_bytes": 935488,
      "prototype_with_anchors_bytes": 528416
    }
  ],
  "totals": {
    "support_vectors": 2000,
    "prototype_vectors": 20,
    "full_vector_bytes": 4096000,
    "full_label_bytes": 16000,
    "prototype_vector_bytes": 40960,
    "prototype_label_bytes": 160,
    "prototype_count_bytes": 160,
    "anchor_bytes": 5242880,
    "full_semantic_bytes": 4112000,
    "prototype_semantic_bytes": 41280,
    "full_with_anchors_bytes": 9354880,
    "prototype_with_anchors_bytes": 5284160,
    "vector_compression": 100.0,
    "semantic_byte_compression": 99.6124031007752,
    "combined_byte_compression": 1.77036274450433
  }
}

| Client/class | Count | Affine mean max error | Affine mean Fro error |
|---|---:|---:|---:|
| 0/4 | 100 | 1.49011612e-08 | 1.49512864e-08 |
| 0/6 | 100 | 3.7252903e-09 | 4.75728967e-09 |
| 1/2 | 100 | 2.38418579e-07 | 1.02444676e-06 |
| 1/6 | 100 | 4.76837158e-07 | 1.13226747e-06 |
| 2/2 | 100 | 2.38418579e-07 | 8.83656583e-07 |
| 2/7 | 100 | 2.38418579e-07 | 1.01804631e-06 |
| 3/3 | 100 | 2.38418579e-07 | 1.12764019e-06 |
| 3/7 | 100 | 3.57627869e-07 | 1.14494253e-06 |
| 4/3 | 100 | 2.38418579e-07 | 1.06433924e-06 |
| 4/5 | 100 | 2.38418579e-07 | 1.1022139e-06 |
| 5/5 | 100 | 2.38418579e-07 | 1.01087676e-06 |
| 5/9 | 100 | 2.38418579e-07 | 1.1264292e-06 |
| 6/0 | 100 | 2.38418579e-07 | 1.05109689e-06 |
| 6/9 | 100 | 2.38418579e-07 | 1.16871433e-06 |
| 7/0 | 100 | 2.38418579e-07 | 9.25144889e-07 |
| 7/8 | 100 | 2.38418579e-07 | 9.82142296e-07 |
| 8/1 | 100 | 2.38418579e-07 | 9.92568289e-07 |
| 8/8 | 100 | 2.38418579e-07 | 8.95987284e-07 |
| 9/1 | 100 | 2.38418579e-07 | 1.17560216e-06 |
| 9/4 | 100 | 2.38418579e-07 | 1.09805978e-06 |

aligned_client_class_prototypes: {"prototype_count": 20, "prototype_hash": "2703c2ae40743bb48a0e21fd43949746ccc9fe85a78b44bdfad006ecc611c97d", "labels_hash": "69ee702dcf18894bda3cbb9355c510766ca73e1bc608d8ef57ed09f304b29f0e", "counts_hash": "66a0ec6b3c34c12ed36340861d0350aa0fe619bae9be9a37892889da248506be", "prototype_dtype": "torch.float32", "label_dtype": "torch.int64", "count_dtype": "torch.int64"}

native_client_class_prototypes_control: {"prototype_count": 20, "prototype_hash": "15c16a0674242edce13a858f0721eb791c9b07a881918d97e63a0b8e9da5a4c5", "labels_hash": "69ee702dcf18894bda3cbb9355c510766ca73e1bc608d8ef57ed09f304b29f0e", "counts_hash": "66a0ec6b3c34c12ed36340861d0350aa0fe619bae9be9a37892889da248506be", "prototype_dtype": "torch.float32", "label_dtype": "torch.int64", "count_dtype": "torch.int64"}
