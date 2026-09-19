# H16-A reuse and increments

Lead24b58c8. Only nested one-owner CIFAR100 seed0, historical order120100,256 anchors,10cycles. Frozen gates from BRIDGE.

Reuse: full_data.allocate/reserve_anchors/preprocessing unchanged; extend cifar100_ownership and CLI owners_per_class default2. Existing H01Client/FedGH/metrics/direct_prototypes unchanged for existing arms. Port minimal FedAvg aggregation from pinned PFLlib serverbase (sample-count weighted full parameters); existing Local CE SGD matches clientAVG and will be directly tested. Evaluate post-aggregation global head and broadcast each subsequent round. No new scientific modules.

Increment1: partition+CLI; files full_data/run/test_cifar100. Prove exact historical default fixture, nested order, disjoint/complete coverage, label-blind anchors, and tiny3arm one-owner entrypoint. Increment2 only after green: narrow FedAvg integration, upstream SGD/aggregation equality and tinyfour-arm paired entrypoint. Finally full suite and remote seed0 representative run. Report adapted from H12 checks with new predeclared coverage gate; frozen H12 comparison. No changes to prior report scripts or prior evidence.

Baseline command: D:/anaconda3/python.exe -m unittest discover -s tests -v; log research_log/h16_baseline_tests.log. Running before edits.

Baseline72PASS220.445s; partition9PASS100.407s; FedAvg focused test log fedavg_tests.log. Both increments green; report_h16a reuses H12 checks with one-owner nested receipts and frozen H16 coverage gate. Full suite next.
