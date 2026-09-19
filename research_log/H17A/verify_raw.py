import json
from pathlib import Path
import numpy as np
from pprtp.tiny_data import index_tiny, TINY_OWNERSHIP_SEED
from pprtp.full_data import allocate, index_hash

root=Path('/home/wenchang/asdasdsad/wjq/PPRTP')
_,labels,_,_,anchors,manifest=index_tiny(root/'shared/tiny-h17/tiny-imagenet-200')
order=np.random.default_rng(TINY_OWNERSHIP_SEED).permutation(200).tolist()
split=allocate(labels,[sorted(order[i::10]) for i in range(10)],anchors,num_classes=200)
receipt={k:v for k,v in manifest.items() if k not in ('train_files','val_files','train_file_hashes','val_file_hashes')}
receipt.update(anchor_indices_sha256=index_hash(anchors),anchor_count=len(anchors),supervised_count=sum(map(len,split['train_indices'])),class_sets=split['class_sets'])
assert receipt['supervised_count']==99744 and receipt['anchor_count']==256
out=root/'research_log/H17A/raw_verification.json'
out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt,indent=2))
