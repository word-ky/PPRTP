"""Matched full-participation FedAvg using pinned PFLlib aggregation unchanged."""
from flcore.servers.serverbase import Server


class MatchedFedAvg(Server):
    def __init__(self,clients):
        self.uploaded_models=[c.model for c in clients]
        total=sum(c.train_samples for c in clients)
        self.uploaded_weights=[c.train_samples/total for c in clients]
        self.aggregate_parameters()
