# H17-A implementation plan

Lead fabcc20: canonical Tiny raw train100000/val10000 (500/50 per200classes); ownership120200/oneowner/20classes per10clients,256 anchors161803 chosen over complete sorted train image list before mapping labels. No official test use. Same frozen methods/readouts and SGD10cycles seed0. ReadinessLocalSeen>=10%; original5Tinygates unchanged.

Reuse map: allocate/reserve_anchors/index_hash from full_data; existing TensorDataset path and every training/readout class; pinned PFLlib generator URL/ToTensor+(.5,.5,.5) normalization and 64CNNdim10816/512base/200head. No augmentation/pretraining. Raw manifest includes classmapping/relativefilenames/fileSHA256/indexSHA and officialvalannotations.

Increment1 after baseline76 green: pprtp/tiny_data.py plus focused tests synthetic directory mapping/validation annotations/label-blind reservation/normalization and oneownerfullcoverage. Do not download until these pass. Increment2: minimal runner dataset/200classes/64dim and existing readout capture/batchaudit generalization; tiny fourarm integration verifies512D/200head/200row readouts/isolation. Then full tests; deploy and actualdata verify/run. Original training math untouched; historical seed0report hashes stayexact.

Remote inventory found no Tiny archive within scoped wjq/home tree. RAM486GiB available; bothA6000~46.46GBused byexistingservice, untouched. Full TensorDataset CPU memory can use existing architecture without adding cache/lazy wrappers. Observed run constraints will determine only minimal operational repair if necessary.

Baseline: python -m unittest discover -s tests -v -> research_log/h17_baseline_tests.log, running before edits. Pinned URL http://cs231n.stanford.edu/tiny-imagenet-200.zip; download only after mappingfixturegreen. Stop if CNNfeature shape mismatch; no substitution.
