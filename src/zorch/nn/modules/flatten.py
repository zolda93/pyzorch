from .module import Module
from ..import functional as F

class Flatten(Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return F.flatten(x)

