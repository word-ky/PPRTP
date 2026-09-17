"""H02-A shared linear head, using the existing PFLlib client means."""
import copy
import torch
import torch.nn.functional as F


def broadcast(head, clients):
    for client in clients:
        client.model.head.load_state_dict(head.state_dict())


def fit_probe(head, clients):
    x=torch.stack([c.protos[k].detach() for c in clients for k in sorted(c.protos)])
    y=torch.tensor([k for c in clients for k in sorted(c.protos)],device=x.device)
    return fit_linear(head,x,y)


def fit_linear(head,x,y,zero=False,max_iter=100,counts=None):
    probe=copy.deepcopy(head)
    x=x.detach()
    if zero:
        with torch.no_grad():
            probe.weight.zero_()
            probe.bias.zero_()
    optimizer=torch.optim.LBFGS(probe.parameters(),line_search_fn='strong_wolfe',
        max_iter=max_iter,tolerance_grad=1e-9,tolerance_change=1e-12)
    def objective(logits):
        return F.cross_entropy(logits,y) if counts is None else weighted_ce(logits,y,counts)
    def score():
        with torch.no_grad():
            logits=probe(x)
            assert torch.isfinite(logits).all()
            return dict(ce=objective(logits).item(),accuracy=(logits.argmax(1)==y).float().mean().item())
    before=score()
    def closure():
        optimizer.zero_grad()
        loss=objective(probe(x))
        loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in probe.parameters())
        return loss
    optimizer.step(closure)
    after=score()
    state=optimizer.state[next(iter(probe.parameters()))]
    info=dict(before=before,after=after,n_iter=state['n_iter'],func_evals=state['func_evals'],
        max_iter=max_iter,line_search_fn='strong_wolfe',tolerance_grad=1e-9,tolerance_change=1e-12,
        lr=1.,weight_norm=probe.weight.norm().item(),bias_norm=probe.bias.norm().item(),
        termination_note='PyTorch exposes iteration/evaluation counts, not an explicit termination reason.')
    assert all(torch.isfinite(p).all() for p in probe.parameters())
    return probe,info


def train_server(head, optimizer, clients):
    order = [(c.id, label) for c in clients for label in sorted(c.protos)]
    x = torch.stack([c.protos[label].detach() for c in clients for label in sorted(c.protos)])
    y = torch.tensor([label for _, label in order], device=x.device)
    assert {id(p) for g in optimizer.param_groups for p in g['params']} == {id(p) for p in head.parameters()}
    def score():
        with torch.no_grad():
            logits = head(x)
            return dict(ce=F.cross_entropy(logits,y).item(), accuracy=(logits.argmax(1)==y).float().mean().item())
    before = score()
    head.train()
    # One deterministic pass: client ID ascending, class ID ascending, batch size 1.
    for row, label in zip(x, y):
        optimizer.zero_grad()
        loss = F.cross_entropy(head(row[None]),label[None])
        loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in head.parameters())
        optimizer.step()
    assert all(torch.isfinite(p).all() for p in head.parameters())
    return dict(before=before, after=score(), sample_order=order, batch_size=1,
                lr=.01, passes=1, weight_norm=head.weight.norm().item(), bias_norm=head.bias.norm().item())


def weighted_ce(logits,y,counts):
    weights=counts.to(logits.dtype)
    return (F.cross_entropy(logits,y,reduction='none')*weights).sum()/weights.sum()
