from zorch import*



def L1Loss(x,target,reduction='mean'):

    assert x.shape == target.shape ,f"target shape {target.shape} must match input shape {x.shape}"

    loss = xp.abs(x.data - target.data)

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    else:
        raise ValueError("{} is not a valid value for reduction".format(reduction))

    return Tensor(loss,x.requires_grad,reduction=reduction)

def L1LossBackward(x,target,child):

    if x.requires_grad:
        reduction = child.kwargs['reduction']
        xgrad = xp.sign(x.data-target.data)

        if reduction == 'sum':
            x.grad += xgrad
        elif reduction == 'mean':
            x.grad += xp.divide(xgrad,x.data.size)

def MSELoss(x,target,reduction='mean'):
    assert x.shape == target.shape ,f"target shape {target.shape} must match input shape {x.shape}"

    loss = (x.data - target.data)**2

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    else:
        raise ValueError("{} is not a valid value for reduction".format(reduction))

    return Tensor(loss,x.requires_grad,reduction=reduction)

def MSELossBackward(x,target,child):

    if x.requires_grad:
        reduction = child.kwargs['reduction']
        xgrad = 2*child.grad*(x.data-target.data)

        if reduction == 'sum':
            x.grad += xgrad
        elif reduction == 'mean':
            x.grad += xp.divide(xgrad,x.data.size)

def BCELoss(x,target,weight=None,reduction='mean'):

    assert x.shape == target.shape

    w = weight.data if weight is not None else 1
    loss = -w * (target.data * xp.clip(xp.log(x.data), -100, None) + (1 - target.data) * xp.clip(xp.log(1-x.data), -100, None))

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    else:
        raise ValueError("{} is not a valid value for reduction".format(reduction))
    
    return Tensor(loss,x.requires_grad,reduction=reduction,weight=w)

def BCELossBackward(x,target,child):

    if x.requires_grad:
        reduction = child.kwargs['reduction']
        weight = child.kwargs['weight']

        xgrad = child.grad * weight * (x.data-target.data) * xp.clip(1 /x.data, None, 1e12) * xp.clip(1 / (1-x.data), None, 1e12)

        if reduction == 'sum':
            x.grad += xgrad
        elif reduction == 'mean':
            x.grad += xp.divide(xgrad,x.data.size)


def BCEWithLogitsLoss(x,target,weight=None,pos_weight=None,reduction='mean'):
    assert x.shape == target.shape


    w = weight.data if weight is not None else 1
    p = pos_weight.data if pos_weight is not None else 1
    log_sigmoid = x.data * (x.data < 0)- xp.log1p(xp.exp(-xp.abs(x.data)))
    loss = w * ((target.data * (1 - p) - 1) * log_sigmoid + x.data - x.data * target.data)

    if reduction == 'mean':
        loss = loss.mean()
    elif reduction == 'sum':
        loss = loss.sum()
    else:
        raise ValueError("{} is not a valid value for reduction".format(reduction))

    return Tensor(loss,x.requires_grad,reduction=reduction,weight=w,pos_weight=p,log_sigmoid=log_sigmoid)

def BCEWithLogitsLossBackward(x,target,child):

    if x.requires_grad:

        log_sigmoid = child.kwargs['log_sigmoid']
        reduction = child.kwargs['reduction']
        w = child.kwargs['weight']
        p = child.kwargs['pos_weight']

        xgrad = child.grad * (w * (target.data * (1 - p) - 1) * xp.exp(-x.data + log_sigmoid) + w * (1 - target.data))

        if reduction == 'sum':
            x.grad += xgrad
        elif reduction == 'mean':
            x.grad += xp.divide(xgrad,x.data.size)

def NLLLoss(x,target,weight=None,ignore_idx=-100,reduction='mean'):

    if not np.issubdtype(target.dtype,np.integer):
        raise RuntimeError(f"expected scalar type int but found{target.dtype},Use dtype=int when  creating target tensor.")

    if weight is None:
        w = xp.ones((1,x.shape[0]),dtype=np.bool)
    else:
        w = weight.data
    dim = x.ndim
    x_data = x.data
    y = target.data

    if dim < 2:
        raise ValueError("Expected 2 or more dimension got {}".format(dim))
    if x_data.shape[0] != y.shape[0]:
        raise ValueError("Expected input batch ({}) to match target batch ({})".format(x_data.shape[0],target.shape[0]))

    if dim == 2: # expand x dim to at least 3
        x_data = x_data[...,None]
    if y.ndim == 1:
        y = y[...,None]

    #if y.shape[1:] != x_data.shape[2:]:
        #raise ValueError("Expected target size {},got{}".format(x_data.shape[2:],y.shape))
    

    ignored = (y != ignore_idx)
    idx = np.indices(y.shape,sparse=True)
    criteria = (idx[0],y,*idx[1:])
    coef = w[0,y]*ignored
    loss = -x_data[criteria]*coef
    N = None

    if reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'mean':
        N = np.count_nonzero(ignored)
        loss = xp.divide(loss.sum(),N,dtype=x.dtype)
    else:
        raise ValueError("{} is not a valid value for reduction".format(reduction))

    return Tensor(loss,x.requires_grad,reduction=reduction,coef=coef,criteria=criteria,N=N)

def NLLLossBackward(x,target,child):

    if x.requires_grad:

        coef = child.kwargs['coef']
        criteria = child.kwargs['criteria']
        N = child.kwargs['N']
        reduction = child.kwargs['reduction']

        if x.ndim == 2:
            x_data = x.data[...,None]

        xgrad = -child.grad * coef

        if reduction == 'mean':
            xgrda = xp.divide(xgrad,N)

        x.grad = xp.zeros_like(x_data)
        x.grad[criteria] = xgrad

        if x.ndim == 2:
            x.grad = x.grad[...,0]


    
def NllLoss(x,target,weight=None,ignore_idx=-100,reduction='mean'):

    if not np.issubdtype(target.dtype,np.integer):
        raise RuntimeError(f'expected scalar type Int but found {target.dtype}. Use "dtype=int" when creating target tensor.')

    if weight is None:
        w = xp.ones((1,x.shape[1]),dtype=np.bool)
    else:
        w = weight.data

    dim = x.ndim

    if dim < 2:
        raise ValueError("Expected 2 or more dimensions (got {})".format(dim))
    if x.shape[0] != target.shape[0]:
        raise ValueError(
            "Expected input batch_size ({}) to match target batch_size ({}).".format(x.shape[0], target.shape[0]))
    if dim == 2:  # expand x dim to at least 3
        x_data = x.data[..., None]
    if target.ndim == 1:  # expand y dim to at least 2
        y = target.data[..., None]
    else:
        y = target.data
    #if y.shape[1:] != x_data.shape[2:]:
        #raise ValueError("Expected target size {}, got {}".format(x_data.shape[2:], y.shape))

    ignored = (y != ignore_idx)
    idx = np.indices(y.shape, sparse=True)
    criteria = (idx[0], y, *idx[1:])
    coef = w[0, y] * ignored
    loss = -x_data[criteria] * coef
    N = None
    if reduction == 'sum':
        loss = loss.sum()
    elif reduction == 'mean':
        N = xp.count_nonzero(ignored)
        loss = xp.divide(loss.sum(), N, dtype=x.dtype)
    elif reduction == 'none':
        pass
    else:
        raise ValueError("{} is not a valid value for reduction".format(reduction))

    return Tensor(loss,x.requires_grad,reduction=reduction,coef=coef,criteria=criteria,N=N)

def NllLossBackward(x,y,child):

    if x.requires_grad:

        coef = child.kwargs['coef']
        criteria = child.kwargs['criteria']
        N = child.kwargs['N']
        reduction = child.kwargs['reduction']

        if x.ndim == 2:
            x_data = x.data[...,None]

        xgrad = -child.grad * coef

        if reduction == 'mean':
            xgrad = xp.divide(xgrad,N)
        x.grad = xp.zeros_like(x_data)

        x.grad[criteria] = xgrad
        if x.ndim == 2:
            x.grad = x.grad[...,0]








