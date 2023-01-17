import zorch as z


def unbroadcast(data,shape):

    #sum all axis that are equal to 1, count from right

    axis0 = tuple(range(data.ndim - len(shape)))
    axis1 = tuple(i - len(shape) for i,v in enumerate(shape) if v == 1)
    data =  z.xp.sum(data,axis=axis0+axis1,keepdims=True)
    data = z.xp.squeeze(data,axis=axis0)
    return data


def Neg(x):
    
    """
    Compute the negative of input

    Parameters
    ----------

    input : x:Tensor

    Result
    ------
    output : -x :Tensor

    """

    return z.Tensor(-x.data,x.requires_grad)

def NegBackward(x,child):

    """
    Compute the gradient of input with respect to a scalar function f

    Parameters
    ----------
    
    input : x:Tensor
    child : z:Tensor
    
    proof
    ----
    x.grad : gradient of f wrt x (df/dx) : ndarray
    child.grad : gradient of f wrt z (df/dz) : ndarray

    dz = -dx
    df/dx = (df/dz)*(dz/dx) = (df/dz)*(-1I) where(1I a matrix of ones)
    df/dx = -df/dz

    """

    if x.requires_grad:
        x.grad += -child.grad

def Add(x,y):

    """
    Compute/Add  two tensor

    Parameters
    ----------
    input0 : x:Tensor
    input1 : y:Tensor

    Results
    -------
    output : z = x+y : Tensor

    """

    return z.Tensor((x.data + y.data),x.requires_grad|y.requires_grad)

def AddBackward(x,y,child):

    """
    Compute the gradient of inputs wrt to a scalar function f

    Parameters
    ----------
    input0 : x:Tensor
    input1 : y:Tensor
    child  : z:Tensor

    proof
    ----
    child.grad  : gradient of f wrt z (df/dz) : ndarray

    dz = dx + dy

    df/dx = (df/dz)*(dz/dx)=(df/dz)*1I
    df/dy = (df/dz)/(dz/dy)=(df/dz)*1I

    """

    if x.requires_grad:
        x.grad += unbroadcast(child.grad,x.shape)
    if y.requires_grad:
        y.grad += unbroadcast(child.grad,y.shape)

def Sub(x,y):

    """
    Compute/Sub two tensor

    Parameters
    ----------
    input0 : x:Tensor
    input1 : y:Tensor

    Result
    ------
    z = x-y :Tensor
    
    """

    return z.Tensor((x.data-y.data),x.requires_grad|y.requires_grad)

def SubBackward(x,y,child):

    """
    Compute the gradient of inputs wrt to a scalar function f

    Parameters
    ----------
    input0 : x:Tensor
    input1 : y:Tensor
    child  : z:Tensor
    child.grad  : gradient of f wrt  z (df/dz) : ndarray

    dz = dx - dy
    df/dx = (df/dz)*(dz/dx)=(df/dz)*(1I)
    df/dy = (df/dz)*(dz/dy)=(df/dz)*(/1I)

    """

    if x.requires_grad:
        x.grad += unbroadcast(child.grad,x.shape)
    if y.requires_grad:
        y.grad -= unbroadcast(child.grad,y.shape)


def Mult(x,y):

    """
    Compute/multiply two tensor

    Parameters
    ----------
    input0 : x:Tensor
    input1 : y:Tensor

    Result
    ------
    output : z=x*y :Tensor

    """

    return z.Tensor(z.xp.multiply(x.data,y.data),x.requires_grad|y.requires_grad)

def MultBackward(x,y,child):

    """
    Compute the gradient of inputs wrt to a scalar function f

    Parameters
    ----------
    input0: x:Tensor
    input1: y:Tensor
    child : z:Tensor

    proof
    -----
    child.grad : gradient of f wrt z

    dz = dx*y + x*dy

    df/dx = (df/dz)*(dz/dx)=(df/dz)*y
    df/dy = (df/dz)*(dz/dy)=(df/dz)*x

    """
    if x.requires_grad:
        x.grad += unbroadcast(child.grad*y.data,x.shape)
    if y.requires_grad:
        y.grad += unbroadcast(child.grad*x.data,y.shape)


def Div(x,y):

    """
    Compute/divide x with y

    Parameters
    ----------
    input0 : x:Tensor
    input1 : y:Tensor

    Result
    ------
    output : z=x/y :Tensor

    """

    return z.Tensor(z.xp.divide(x.data,y.data),x.requires_grad|y.requires_grad)

def DivBackward(x,y,child):

    """
    dz = (dx*y - x*dy)/y^2

    df/dx = (df/dz)*(dz/dx) = (df/dz)/y
    df/dy = (df/dz)*(dz/dy) = -(df/dz)*x/y^2

    """

    if x.requires_grad:
        x.grad += unbroadcast(z.xp.divide(child.grad,y.data),x.shape)
    if y.requires_grad:
        y.grad -= unbroadcast(child.grad * x.data/(y.data * y.data),y.shape)


def Pow(x,y):

    """
    z = x^y

    """
    return z.Tensor(z.xp.power(x.data,y.data),x.requires_grad|y.requires_grad)

def PowBackward(x,y,child):

    """
    dz = dx*y*x^(y-1) + log(x)*z*dy
    df/dx = (df/dz)*(dz/dx) = (df/dz)*y*x^(y-1)
    df/dy = (df/dz)/(dz/dy) = (df/dz)*log(x)*z

    """

    if x.requires_grad:
        x.grad += child.grad * y.data*z.xp.power(x.data,y.data-1)
    if y.requires_grad:
        y.grad += child.grad * z.xp.log(x.data) * child.data

def Exp(x):

    """
    z = exp(x)

    """

    return z.Tensor(z.xp.exp(x.data),x.requires_grad)

def ExpBackward(x,child):

    """
    dz = dx*exp(x)
    df/dx = (df/dz)*(dz/dx) = (df/dz)*exp(x) = (df/dz)*z

    """

    if x.requires_grad:
        x.grad += child.grad*child.data 


"""def matmul_broadcast(a: np.ndarray, b: np.ndarray):
    assert a.ndim != 0 and b.ndim != 0

    def broadcast(a: np.ndarray, b: np.ndarray):
        if a.ndim < b.ndim:
            b, a = broadcast(b, a)
            return a, b
        for _ in range(a.ndim - b.ndim):
            b = b[np.newaxis, :]
        shape_a = a.shape
        shape_b = b.shape
        for i in range(3, b.ndim + 1): # Here is to broadcast the dimensions after 2 dimensions
            if shape_a[-i] != shape_b[-i] and shape_a[-i] != 1 and shape_b[
                    -i] != 1:
                return None
            elif shape_a[-i] == 1:
                a = np.repeat(a, shape_b[-i], -i)
            elif shape_b[-i] == 1:
                b = np.repeat(b, shape_a[-i], -i)
        assert a.shape[:-2] == b.shape[:-2]
        return a, b

    if a.ndim == 1 and b.ndim == 1:
        return a @ b
    elif a.ndim == 1:
        a = a[np.newaxis, :]
        a, b = broadcast(a, b)
        c = a @ b
        return c.reshape(c.shape[:-2] + c.shape[-1:])
    elif b.ndim == 1:
        b = b[:, np.newaxis]
        a, b = broadcast(a, b)
        c = a @ b
        return c.reshape(c.shape[:-1])
    else:
        a, b = broadcast(a, b)
        return a @ b"""

def Mm(x,y):

    """
    z = matmul(x,y)

    """

    if x.ndim == 0 or y.ndim == 0:
        raise RuntimeError(f"both inputs need to be at least 1D,but they are {x.ndim}D and {y.ndim}D")
    
    x_low = x.ndim < 2
    y_low = y.ndim < 2

    x_shape = (1,) + x.shape if x_low else x.shape
    y_shape = y.shape + (1,) if y_low else y.shape

    if x.shape[-1] != y.shape[-2]:
        raise ValueError(f"Dimension mismatch:input0 has a shape {x.shape} ans input1 hase a shape of {y.shape}")

    return z.Tensor(x.data @ y.data,x.requires_grad|y.requires_grad,x_low=x_low,y_low=y_low)

def MmBackward(x,y,child):

    """
    let Y = AXB where A,B,X are matrices
    dY = dAXB + AdXB + AXdB

    take A,B constants we got :dY = AdXB
    Y.T : transpose of matrix Y

    df = trace((df/dY).T dY) = trace((df/dY).T *AdXB) = trace(((A.T)(df/dY)(B.T)).T.dX)

    take A = 1I -> df = trace(((df/dY).B.T).dX)-> df/dx = (df/dY).B.T
    take B = 1I -> df = trace((A.T.(df/dY).dX) ->df/dY  = A.T.(df/dY)

    z = matmu(x,y)

    df/dx = (df/dz).(y.T)
    df/dy = (x.T).(df/dz)

    """

    x_low = child.kwargs['x_low']
    y_low = child.kwargs['y_low']

    grad = child.grad

    if x_low:
        grad = grad[...,None]
        x_data = x.data[:,None]
    else:
        x_data = x.data

    if y_low:
        grad = grad[...,None,:]
        y_data = y.data[None]
    else:
        y_data = y.data

    if x.requires_grad:
        dx = grad @ z.xp.swapaxes(y_data,-1,-2) # refer to (df/dx)
        x.grad += unbroadcast(dx,x.shape)
    if y.requires_grad:
        dy = z.xp.swapaxes(x_data,-1,-2) @ grad # refer to (df/dy)

        if y_low:
            y.grad += z.xp.sum(dy,axis=tuple(range(dy.ndim - y.ndim)))[...,0]
        else:
            y.grad += unbroadcast(dy,y.shape)


def Transpose(x,axes):

    return z.Tensor(z.xp.transpose(x.data,axes=axes),x.requires_grad,axes=axes)

def TransposeBackward(x,child):

    if x.requires_grad:
        axes = child.kwargs['axes']
        if axes is None:
            x.grad += z.xp.transpose(child.grad)
        else:
            x.grad += z.xp.transpose(child.grad,axes=list(axes).sort())


def Sum(x,axis,keepdims):
    
    if axis is None:
        axis = tuple(range(x.ndim))
    if isinstance(axis,int):
        axis = (axis,)

    return z.Tensor(z.xp.sum(x.data,axis=axis,keepdims=keepdims),x.requires_grad,axis=axis,keepdims=keepdims)

def SumBackward(x,child):

    if x.requires_grad:

        axis = child.kwargs['axis']
        keepdims = child.kwargs['keepdims']
        grad = child.grad

        if not keepdims:
            grad = z.xp.expand_dims(grad,axis=axis)

        strides = list(grad.strides)

        for i in axis:
            strides[i] = 0

        x.grad += z.xp.lib.stride_tricks.as_strided(grad,shape=x.shape,strides=strides)

def Mean(x,axis,keepdims):
    
    if axis is None:
        axis = tuple(range(x.ndim))
    if isinstance(axis,int):
        axis = (axis,)
    return z.Tensor(z.xp.mean(x.data,axis=axis,keepdims=keepdims),x.requires_grad,axis=axis,keepdims=keepdims)


def MeanBackward(x,child):

    if x.requires_grad:

        axis = child.kwargs['axis']
        keepdims = child.kwargs['keepdims']
        grad = child.grad
        N = 1

        if not keepdims:
            grad = z.xp.expand_dims(grad,axis=axis)

        strides = list(grad.strides)

        for i in axis:
            N *= x.shape[i]
            strides[i] = 0

        x.grad += z.xp.lib.stride_tricks.as_strided(z.xp.divide(grad,N,dtype=grad.dtype),shape=x.shape,strides=strides)


def Var(x,axis,ddof,keepdims):

    if axis in None:
        axis = tuple(range(x.ndim))
    if isinstance(axis,int):
        axis = (axis,)
    return z.Tensor(z.xp.var(x.data,dtype=x.dtype,axis=axis,ddof=ddof,keepdims=keepdims),x.requires_grad,axis=axis,ddof=ddof,keepdims=keepdims)

def VarBackward(x,child):

    if x.requires_grad:

        axis = child.kwargs['axis']
        ddof = child.kwargs['ddof']
        keepdims = child.kwargs['keepdims']
        grad = child.grad

        if not keepdims:
            grad = z.xp.expand_dims(grad,axis=axis)
        N = 1
        _mean = z.xp.mean(x.data,axis=axis,keepdims=True)

        for i in axis:
            N *= x.shape[i]
        x.grad += 2 * grad * z.xp.divide(x.data - _mean,N-ddof,dtype=grad.dtype)


def Concat(xs,axis):

    requires_grad = any(x.requires_grad for x in xs)
    data = tuple(x.data for x in xs)
    indices = z.np.cumsum([x.shape[axis] for x in xs])
    value = z.xp.concatenate(data,axis)
    return z.Tensor(value,requires_grad,axis=axis,indices=indices)

def ConcatBackward(xs,child):

    axis = child.kwargs['axis']
    indices = child.kwargs['indices']
    grad = z.xp.split(child.grad,indices_or_sections=indices,axis=axis)

    for idx ,x in enumerate(xs):
        if x.requires_grad:
            x.grad += grad[idx]

def Slice(x,key):

    return z.Tensor(x.data[key],x.requires_grad,key=key)

def SliceBackward(x,child):

    if x.requires_grad:

        key = child.kwargs['key']
        x.grad[key] += child.grad


def View(x,shape):

    return z.Tensor(z.xp.reshape(x.data,shape),x.requires_grad)

def ViewBackward(x,child):

    if x.requires_grad:
        x.grad += z.xp.reshape(child.grad,x.shape)


def Repeat(x,reps):

    return z.Tensor(z.xp.tile(x.data,reps),x.requires_grad,reps=reps)

def RepeatBackward(x,child):

    if x.requires_grad:

        reps = child.kwargs['reps']
        new_shape=[]
        sum_axes = []
        idx =0

        for i in range(-len(reps),0):
            new_shape.append(reps[i])
            sum_axes.append(idx)
            idx+=1
        else:
            if reps[i] == 1:
                new_shape.append(x.shape[i])
                idx+=1
            else:
                new_shape.extend([reps[i],x.shape[i]])
                sum_axes.append(idx)
                idx+=2
        x.grad += child.grad.reshape(new_shape).sum(tuple(sum_axes))


def Expand(x,*dims):

    leading_dim = len(dims)-len(x.shape)
    dims = z.np.array(dims)
    x_shape = z.np.array((1,)*lending_dim + x.shape)
    singleton = z.np.logical_and(x_shape==1,dims>1)
    dims[~singleton]=x_shape[~singleton]
    strides = z.np.array((0,)*leading_dim + x.data.strides)
    strides[~singleton] = 0
    value = z.xp.lib.stride_tricks.as_strided(x.data,shape=dims,strides=strides)
    return z.Tensor(value,x.requires_grad,sum_axes=tuple(t.xp.arange(len(dims))[singleton]),leading_dim=tuple(range(leading_dim)))


def ExpandBackward(x,child):

    if x.requires_grad:

        sum_axes=child.kwargs['sum_axes']
        leading_dim=child.kwargs['leading_dim']
        x.grad += child.grad.sum(axis=sum_axes,keepdims=True).squeeze(leading_sim)


def Squeeze(x,axis=None):

    if axis is None:
        axis = tuple(range(x.ndim))
    if isinstance(axis,int):
        axis = (axis,)

    unchanged = False

    axis = tuple(i for i in axis if x.shape[i] ==1)

    if len(axis) == 0:
        value = x.data.copy()
        unchanged=True
    else:
        value = z.xp.squeeze(x.data,axis=axis)

    return z.Tensor(value,x.requires_grad,axis=axis,unchanged=unchanged)


def SqueezeBackward(x,child):

    if x.requires_grad:

        axis=child.kwargs['axis']
        unchanged=child.kwargs['unchanged']

        if unchanged:
            x.grad += child.grad
        else:
            x.grad += z.xp.expand_dims(child.grad,axis=axis)
    

def Unsqueeze(x,axis):

    value = z.xp.expand_dims(x.data,axis=axis)
    return z.Tensor(value,x.requires_grad,axis=axis)

def UnsqueezeBackward(x,child):

    if x.requires_grad:

        axis = child.kwargs['axis']

        x.grad += z.xp.squeeze(child.grad,axis=axis)


def MaskedFill(x,mask,val):

    value = x.data.copy()
    mask = z.xp.lib.stride_tricks.as_strided(mask.data, shape=value.shape,strides=(0,) * (value.ndim - mask.ndim) + mask.strides)

    value[mask] = val
    return z.Tensor(value,x.requires_grad,mask=mask)


def MaskedFillBackward(x,child):

    if x.requires_grad:

        mask = ~child.kwargs['mask']
        x.grad[mask] += child.grad[mask]
        #value = z.xp.zeros_like(x.data)
        #value[mask] = child.grad[mask]
        #x.grad += value








    
