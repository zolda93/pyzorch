from zorch import*

def Pad(x,padding,mode,value):
    assert len(padding) % 2 == 0,"padding lenght must be divisible by 2"
    assert len(padding)//2 <= x.ndim ,"padding lenght too large"

    padding_dims = len(padding) //2
    pad_with = tuple((0,0) if i >= padding_dims else padding[(2 * i):(2 * i + 2)] for i in reversed(range(x.ndim)))

    if mode == 'constant':
        value = xp.pad(x.data,pad_width,mode=mode,constant_values = value)
    else:
        raise
    return Tensor(value,x.requires_grad,pad_with=pad_with,mode=mode)

def PadBackward(x,child):
    if x.requires_grad:
        pad_width = child.kwargs['pad_width']
        border = tuple(slice(pad_width[i][0],pad_width[i][0] + x.shape[i]) for i in range(x.ndim))
        x.grad += child.grad[border]


