import torch
import torch.nn as nn
import torch.nn.functional as F

semantic_pred=torch.tensor([[0.9613, 0.0539, 0.3718],[0.9741, 0.0274, 0.8749],[0.9850, 0.0180, 0.7452]])
semantic_log=torch.tensor([[-0.6720, -1.5794, -1.2615],[-0.8301, -1.7768, -0.9293],[-0.7734, -1.7404, -1.0131]])
label=torch.tensor([0, 0, 0])

func1=nn.Softmax(dim=-1)
semantic=torch.log(func1(semantic_pred))
print(semantic)

cross_entropy_loss1 = F.nll_loss(
semantic_log,
label.long()
)

criterion=nn.CrossEntropyLoss()
cross_entropy_loss2 = criterion(semantic_pred,label)

print(cross_entropy_loss1)
print(cross_entropy_loss1)