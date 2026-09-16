# H01 execution contract — 2026-09-16

User requires Jianqing Zhang's PFLlib. Official upstream is TsingZ0/PFLlib,
pinned submodule 0169ba7e412c9856a08bb3faefab1e35f538a3c1 (Apache-2.0).
No prior implementation exists in PPRTP. No other project's modified code is imported.

Reuse map: FedAvgCNN/BaseHeadSplit unchanged; clientProto/Client initialization,
SGD, model ownership, prototype collection/agg_func, set_protos retained.
Narrow client subclass adapts the single training loop for selectable injection,
sample counts and diagnostics. PFLlib prototype aggregation is unweighted across
clients; H01 explicitly requires sample weighting, applied identically to both arms.
PFLlib CIFAR generator merges official train/test before splitting and tests only
local classes. H01 needs official train/test separation and all-class test loaders;
these data/evaluation seams are adapted and documented, not silently inherited.

Increments: (1) unmodified upstream client two-round synthetic health test;
(2) injection and weighted aggregation with gradient/parity tests;
(3) fixed CIFAR split, common evaluation and paired three-mode smoke;
(4) three-seed real-data mechanism run. Each increment tested before the next.

Predeclared tiny real-data protocol: seeds 0,1,2; 10 clients; cyclic two-class
sets after seeded class permutation, each class owned by two clients; disjoint
training partition, 100 training examples per client/class (2000 total),
100 official test examples/class (1000 total) shared for evaluation only.
PFLlib FedAvgCNN 512-D feature, identical initialization across clients/arms,
SGD lr .01, no momentum/decay, batch 32, 1 local epoch, 10 rounds, full participation,
lambda=1, GPC scale=10. No augmentation; upstream normalization .5/.5.
This is a bounded CIFAR-10 subset mechanism test, not benchmark reproduction.

Raw feature prototypes collected during training exactly as PFLlib; sample-count
weighted aggregation of observed representations, no EMA. Exclude invalid classes;
first round has no global loss. Same seed, batch order, construction and aggregation
RULE across arms; prototype values naturally differ as the models learn.

Evaluate every arm using the same three rules: local head, raw nearest-prototype
L2 (upstream FedProto inference), normalized cosine prototype classifier. Primary
injection comparison uses common cosine inference; report common L2 and head as
controls. Local prototype readouts are diagnostic server aggregation only, never
used in local training; local-only deployment metric is its head accuracy.
Do not attribute an inference-rule change to the training objective.

Remote root /home/wenchang/asdasdsad/wjq/PPRTP created and writable.
Existing TTFL environment Torch2.4.0+cu121/torchvision0.19.0+cu121 runs A6000
CUDA tensors successfully despite NVML mismatch. Reuse environment read-only.
