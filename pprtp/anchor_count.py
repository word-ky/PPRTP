"""H04-A fixed nested prefixes of the previously frozen anchor ordering."""
import hashlib,json
from torch.utils.data import TensorDataset
from pprtp.paired import analyze_paired


COUNTS=(1000,512,256,128,64)


def prefix(anchors,indices,n):
    selected=indices[:n]
    return TensorDataset(*(t[:n] for t in anchors.tensors)),dict(indices=selected,
        indices_sha256=hashlib.sha256(json.dumps(selected,separators=(',',':')).encode()).hexdigest())


def historical_fields(result):
    return {k:v for k,v in result.items() if k not in ('rank_diagnostics','existing_gradients_unchanged')}


def analyze_counts(clients,head,anchors,support,test,tensor_hash,metrics,indices,historical):
    arms={}
    for n in COUNTS:
        subset,receipt=prefix(anchors,indices,n)
        result=analyze_paired(clients,head,subset,support,test,tensor_hash,metrics,
            max_iter=2000,audit=True,rank_diagnostics=True)
        if n==1000:
            assert historical_fields(result)==historical, 'H04-A invalid: N1000 reproduction failed'
        assert result['state_before']==historical['state_before']
        arms[str(n)]=dict(result=result,anchor_receipt=receipt,
            payload_bytes_per_client=n*512*4,payload_bytes_total=10*n*512*4,
            compression_factor=1000/n)
        print(f'H04-A N={n} complete; support_fit={result["fit"]["after"]["accuracy"]}',flush=True)
    return dict(arms=arms,n1000_exact=True)
