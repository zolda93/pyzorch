from .module import Module
from .. import functional as F

class L1Loss(Module):
    def __init__(self,reduction='mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self,x,target):
        return F.l1_loss(x,target,reduction=self.reduction)

class MSELoss(Module):
    def __init__(self,reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self,x,target):
        return F.mse_loss(x,target,reduction=self.reduction)

class BCELoss(Module):
    def __init__(self,reduction='mean',weight=None):
        super().__init__()
        self.reduction = reduction
        self.weight = weight

    def forward(self,x,target):
        return F.binary_cross_entropy(x,target,weight=self.weight,reduction=self.reduction)

class BCEWithLogitsLoss(Module):
    def __init__(self,reduction='mean',weight=None,pos_weight=None):
        super().__init__()
        self.reduction = reduction
        self.weight = weight
        self.pos_weight = pos_weight

    def forward(self,x,target):
        return F.binary_cross_entropy_with_logits(x,target,weight=self.weight,pos_weight=self.pos_weight,reduction=self.reduction)

class NLLLoss(Module):
    def __init__(self,weight=None,ignore_idx=-100,reduction='mean'):
        super().__init__()
        self.weight = weight
        self.ignore_idx = ignore_idx
        self.reduction = reduction

    def forward(self,x,target):
        return F.nll_loss(x, target, weight=self.weight, ignore_idx=self.ignore_idx, reduction=self.reduction)

class CrossEntropyLoss(Module):
    def __init__(self,weight=None,ignore_idx=-100,reduction='mean'):
        super().__init__()
        self.weight = weight
        self.ignore_idx = ignore_idx
        self.reduction = reduction

    def forward(self,x,target,axis=1):
        return F.cross_entropy_loss(x,target,axis=axis,weight=self.weight,ignore_idx=self.ignore_idx,reduction=self.reduction)
