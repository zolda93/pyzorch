from zorch import*

def Dropout(x,p,training,inplace):
    if p < 0.0 or p > 1.0:
        raise ValueError("dropout probability hase to be between 0 and 1,got {}".format(p))

    if training and p > 0.0:
        mask = xp.random.binomial(1,1-p,size=x.shape)
        if inplace:
            x.data *= mask
            if p!= 1.:
                x.data /= 1-p
            value = x.data
        else:
            value = xp.multiply(x.data,mask)
            if p != 1.:
                value /= 1-p
        return Tensor(value,x.requires_grad,p=p,mask=mask)
    else:
        return x

def DropoutBackward(x,child):
    if x.requires_grad:
        p = child.kwargs['p']
        mask = child.kwargs['mask']
        x.grad += xp.multiply(child.grad,mask) / (1-p*(p!=1.))


