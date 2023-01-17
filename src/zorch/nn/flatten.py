from zorch import*

def Flatten(x):
    return Tensor(x.data.reshape(x.shape[0],-1),x.requires_grad)

def FlattenBackward(x,child):
    if x.requires_grad:
        x.grad+= child.grad.reshape(x.shape)
