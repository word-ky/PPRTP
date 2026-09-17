"""H02-A shared linear head, using the existing PFLlib client means."""
import torch
import torch.nn.functional as F


def broadcast(head, clients):
    for client in clients:
        client.model.head.load_state_dict(head.state_dict())


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
