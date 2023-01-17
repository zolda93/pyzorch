from zorch import*

def Linear(x,w,bias):
    requires_grad = x.requires_grad | w.requires_grad

    value = x.data @ w.data.T

    if bias is not None:
        requires_grad |= bias.requires_grad
        value += bias.data
    return Tensor(value,requires_grad,bias=bias)

def LinearBackward(x,w,child):

    bias = child.kwargs['bias']

    if bias is not None:
        if bias.requires_grad:
            bias.grad += child.grad.sum(tuple(range(child.grad.ndim-1)))
    if x.requires_grad:
        x.grad += child.grad@w.data
    if w.requires_grad:
        w.grad += xp.tensordot(child.grad,x.data,axes=(np.arange(child.grad.ndim - 1),np.arange(child.grad.ndim - 1)))



