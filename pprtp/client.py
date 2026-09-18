"""Adapted from PFLlib clientproto.py (Apache-2.0); see PROVENANCE.md."""
from collections import defaultdict
import time
import hashlib
import torch
import torch.nn.functional as F
from flcore.clients.clientproto import clientProto, agg_func


def prototype_bank(protos, classes, reference):
    bank = reference.new_zeros(classes, reference.shape[-1])
    valid = torch.zeros(classes, dtype=torch.bool, device=reference.device)
    for label, value in (protos or {}).items():
        assert 0 <= label < classes
        bank[label] = value.detach()
        valid[label] = True
    return bank.detach(), valid


def knowledge_loss(z, y, bank, valid, mode, scale=10., class_set=None):
    bank = bank.detach()
    if mode == "gpc_seen_match":
        seen = torch.zeros_like(valid)
        seen[class_set] = True
        valid = valid & seen
    if mode in ("local", "fedgh") or not valid.any():
        return z.sum() * 0
    assert ((y >= 0) & (y < len(bank))).all()
    eligible = valid[y]
    if mode == "fedproto":
        # Exactly upstream MSE averaging, including zero loss for unavailable targets.
        target = z.detach().clone()
        target[eligible] = bank[y[eligible]]
        return F.mse_loss(z, target)
    if mode in ("gpc", "gpc_all_match", "gpc_seen_match"):
        if not eligible.any():
            return z.sum() * 0
        logits = scale * F.normalize(z[eligible], dim=1) @ F.normalize(bank, dim=1).T
        return F.cross_entropy(logits.masked_fill(~valid[None], -torch.inf), y[eligible])
    raise ValueError(mode)


def aggregate(clients):
    """Weight by the actual class representation count used by upstream agg_func."""
    totals, counts = {}, defaultdict(int)
    for client in clients:
        for label, proto in client.protos.items():
            n = client.proto_counts[label]
            totals[label] = totals.get(label, torch.zeros_like(proto)) + n * proto.detach()
            counts[label] += n
    return {c: (total / counts[c]).detach() for c, total in totals.items()}


def denominator_direction(z, y, bank, valid, scale, class_set):
    """Both derivatives on the exact same pre-update feature tensor and bank."""
    gradients = [torch.autograd.grad(knowledge_loss(z, y, bank, valid, mode, scale, class_set),
                                     z, retain_graph=True)[0].flatten()
                 for mode in ('gpc_all_match', 'gpc_seen_match')]
    all_grad, seen_grad = gradients
    return dict(cosine=F.cosine_similarity(all_grad[None], seen_grad[None]).item(),
                all_norm=all_grad.norm().item(), seen_norm=seen_grad.norm().item(),
                unscaled_all_seen_norm_ratio=(all_grad.norm()/seen_grad.norm()).item())


class H01Client(clientProto):
    def __init__(self, args, *a, **kw):
        super().__init__(args, *a, **kw)
        self.mode = getattr(args, "mode", "fedproto")
        self.scale = getattr(args, "scale", 10.)

    def train(self):
        # Same representation collection timing, optimizer and local epochs as upstream.
        start = time.time()
        self.model.train()
        protos = defaultdict(list)
        local_sum = knowledge_sum = samples = 0
        self.diagnostic = None
        online=self.mode in ("pprtp_all_lag1","pprtp_seen_lag1")
        if online: self.batch_hashes=[]
        for _ in range(self.local_epochs):
            for x, y in self.load_train_data():
                x, y = x.to(self.device), y.to(self.device)
                if online: self.batch_hashes.append(hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()+y.detach().cpu().contiguous().numpy().tobytes()).hexdigest())
                z = self.model.base(x)
                local = self.loss(self.model.head(z), y)
                bank, valid = prototype_bank(self.global_protos, self.num_classes, z)
                lag=getattr(self,'aligned_bank',None) if online else None
                if lag is not None:
                    from pprtp.online import aligned_loss,gradient_direction
                    bank=lag;valid=torch.ones(len(bank),dtype=torch.bool,device=z.device)
                    knowledge=aligned_loss(z,y,bank,self.aligned_transform,self.class_set,self.mode=='pprtp_all_lag1',self.scale)
                else:
                    knowledge = knowledge_loss(z, y, bank, valid, 'fedgh' if online else self.mode, self.scale,
                                               getattr(self, "class_set", None))
                if self.diagnostic is None and self.id == 0:
                    grad = torch.autograd.grad(knowledge, tuple(self.model.base.parameters()),
                                               retain_graph=True, allow_unused=True)
                    norm = sum(g.square().sum() for g in grad if g is not None).sqrt()
                    local_grad = torch.autograd.grad(local, tuple(self.model.base.parameters()),
                                                     retain_graph=True)
                    local_norm = sum(g.square().sum() for g in local_grad).sqrt()
                    scaled_norm = abs(self.lamda) * norm
                    missing = torch.ones(self.num_classes, dtype=torch.bool, device=z.device)
                    missing[getattr(self, "class_set", [0, 1])] = False
                    mass = None
                    if valid.any() and not online:
                        logits = self.scale * F.normalize(z.detach(), dim=1) @ F.normalize(bank, dim=1).T
                        probs = logits.masked_fill(~valid[None], -torch.inf).softmax(1)
                        mass = probs[:, missing].sum(1).mean().item()
                    self.diagnostic = dict(feature_extractor_knowledge_grad_norm=norm.item(),
                        local_grad_norm=local_norm.item(), scaled_knowledge_grad_norm=scaled_norm.item(),
                        knowledge_local_grad_ratio=(scaled_norm/local_norm).item() if local_norm.item() else None,
                        missing_probability_on_seen=mass, valid_mask=valid.tolist())
                    if online and lag is not None:
                        direction=gradient_direction(z,y,bank,self.aligned_transform,self.class_set,self.scale)
                        self.diagnostic['denominator_feature_gradients']=direction
                        self.diagnostic['missing_probability_on_seen']=direction['missing_probability_mass']
                        self.diagnostic['preupdate_feature_hash']=hashlib.sha256(z.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
                    elif valid.any():
                        self.diagnostic['denominator_feature_gradients'] = denominator_direction(
                            z, y, bank, valid, self.scale, self.class_set)
                if online: assert torch.isfinite(local) and torch.isfinite(knowledge)
                for j, label in enumerate(y.tolist()):
                    protos[label].append(z[j].detach())
                self.optimizer.zero_grad()
                (local + self.lamda * knowledge).backward()
                self.optimizer.step()
                samples += len(y)
                local_sum += local.item() * len(y)
                knowledge_sum += knowledge.item() * len(y)
        self.proto_counts = {label: len(values) for label, values in protos.items()}
        self.protos = agg_func(protos)
        self.losses = dict(local=local_sum/samples, knowledge=knowledge_sum/samples)
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start
