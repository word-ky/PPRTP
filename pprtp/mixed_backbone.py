"""H13-A fixed even/odd PFLlib backbone construction; no adapter."""
import torch
from flcore.trainmodel.models import FedAvgCNN,BaseHeadSplit
from flcore.trainmodel.resnet import resnet18


def build_mixed(seed,num_classes,tensor_hash):
    torch.manual_seed(seed)
    models=[];receipts=[]
    for i in range(10):
        name='FedAvgCNN' if i%2==0 else 'ResNet18'
        base=FedAvgCNN(in_features=3,num_classes=num_classes,dim=1600) if i%2==0 else resnet18(num_classes=num_classes)
        head=base.fc;base.fc=torch.nn.Identity();model=BaseHeadSplit(base,head)
        before=tensor_hash(model.state_dict().values())
        model.eval()
        with torch.no_grad():z=model.base(torch.zeros(2,3,32,32))
        model.train()
        assert z.shape==(2,512) and head.weight.shape==(num_classes,512) and head.bias.shape==(num_classes,)
        assert tensor_hash(model.state_dict().values())==before
        receipts.append(dict(client=i,architecture=name,initial_model_hash=before,
            initial_base_hash=tensor_hash(base.state_dict().values()),initial_head_hash=tensor_hash(head.state_dict().values()),
            feature_dim=512,head_shapes={k:list(v.shape) for k,v in head.state_dict().items()},
            parameter_count=sum(p.numel() for p in model.parameters()),
            buffer_names=list(dict(model.named_buffers())),pretrained=False))
        models.append(model)
    return models,receipts
