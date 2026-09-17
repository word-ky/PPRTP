# H03-A paired-anchor orthogonal alignment

Anchor hash `1dd91744e595c7eb36449cd1a1ad362ac9b4d42def0b30dd14707c74463a1125`.1000 train-only images, seed161803,label-blind selection; no train/oracle/support overlap.
Exact H02-E support reused; online H02-A round1/2 exact; state/modes/CPU-CUDA RNG unchanged.

| Round | Seen % | Missing % | All % | Macro % |
|---|---:|---:|---:|---:|
| 2 | 31.650000 | 29.387500 | 29.840000 | 29.840000 |

Head CE 2.30258393 → 0.338887542; train accuracy 0.100000 → 0.890000; LBFGS 100 iterations/106 evaluations.
q_align=0.934236747; head adequate=False. Fixed missing references:.0875% owner-support,31.45% shared oracle.

| Client | Raw residual before | Centered before | Centered after | Relative reduction | Orthogonality(double) | Orthogonality(applied float32) |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 28.891671 | 10.285769 | 5.54559167 | 0.460848121 | 4.67111108e-12 | 3.9640604e-06 |
| 2 | 41.7452335 | 13.9641321 | 6.42352745 | 0.539998088 | 4.00768455e-12 | 4.23783422e-06 |
| 3 | 42.2130462 | 13.4967284 | 6.27263262 | 0.53524792 | 4.12362153e-12 | 4.21841469e-06 |
| 4 | 42.2848091 | 13.8878737 | 6.58414579 | 0.525906851 | 4.26945981e-12 | 4.27801797e-06 |
| 5 | 48.9793557 | 15.4092138 | 7.4132689 | 0.51890674 | 4.17107576e-12 | 4.2412421e-06 |
| 6 | 42.9499595 | 13.9383887 | 7.17568979 | 0.485185128 | 4.22168342e-12 | 4.25488588e-06 |
| 7 | 38.2432447 | 12.787637 | 7.06541108 | 0.447481105 | 4.34397317e-12 | 4.06557683e-06 |
| 8 | 44.7151605 | 14.6960758 | 7.64979178 | 0.479467044 | 4.23052704e-12 | 4.336031e-06 |
| 9 | 31.1931249 | 10.6878213 | 5.50372154 | 0.485047384 | 4.72363225e-12 | 3.96534006e-06 |

Residuals are Frobenius norms; relative reduction compares centered pre/post residuals. Client0 is identity (zero residual).
Raw JSON retains per-client class counts/correct counts, transform hashes, head norms/hash, and side-effect receipts.
