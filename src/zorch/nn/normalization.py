from zorch import*

def LayerNorm(x,weight,normalized_shape,bias,eps):
    if weight is not None:
        if weight.shape != normalized_shape:
            raise RuntimeError(f"Expected weight to be of same shape as normalized_shape, but got weight of shape {weight.shape} and normalized_shape = {normalized_shape}")

    if bias is not None:
        if bias.shape != normalized_shape:
            raise RuntimeError(f"Expected bias to be of same shape as normalized_shape, but got bias of shape {bias.shape} and normalized_shape = {normalized_shape}")

    x_data = x.data
    axis = tuple(-i for i in range(1,1+len(normalized_shape)))
    shape = x_data.shape[-len(normalized_shape):]
    mean = xp.mean(x_data,axis=axis,keepdims=True)
    var = xp.var(x_data,axis=axis,keepdims=True)
    std = xp.sqrt(var + eps)

    requires_grad = x.requires_grad

    norm = (x_data - mean)/std
    value = norm

    if weight is not None:
        value *= weight.data
        requires_grad |= weight.requires_grad

    if bias is not None:
        value += bias.data
        requires_grad |= bias.requires_grad

    return Tensor(value,requires_grad,bias=bias,axis=axis,shape=shape,norm=norm,mean=mean,var=var+eps,std=std)


def LayerNormBackward(x,w,child):

    axis = child.kwargs['axis']
    shape = child.kwargs['shape']
    norm = child.kwargs['norm']
    mean = child.kwargs['mean']
    var = child.kwargs['var']
    std = child.kwargs['std']
    bias = child.kwargs['bias']

    sum_axis = tuple(range(child.grad.ndim - len(axis)))

    if w is not None:
        if w.requires_grad:
            w.grad += (norm * child.grad).sum(sum_axis)
    if bias is not None:
        if bias.requires_grad:
            bias.grad += child.grad.sum(sum_axis)

    if x.requires_grad:
        grad = child.grad
        if w is not None:
            grad = grad * w.data
        N = np.prod(shape)
        a = xp.multiply(N,grad,dtype=grad.dtype) - grad.sum(axis=axis,keepdims=True)
        b = (x.data - mean) / var * xp.sum(grad * (x.data - mean),axis=axis,keepdims=True)
        value = xp.divide(a-b,N,dtype=grad.dtype) / std
        x.grad += value



def BatchNorm(x,weight,running_mean,running_var,bias,training,momentum,eps):
    def expand_dim(*dims):
        extra_dim = x.ndim - 2
        return tuple(i.reshape(i.shape + (1,)*extra_dim) for i in dims)

    def check_batch_size(size):
        size_prods = size[0]
        for i in range(len(size) - 2):
            size_prods *= size[i+2]
        if size_prods == 1:
            raise ValueError(f"Expected more than 1 value per channel when training, got input size {size}")

    axis = (0,) + tuple(range(2,x.ndim))
    shape = (x.shape[0],) + x.shape[2:]

    if training:
        check_batch_size(x.shape)
        mean = xp.mean(x.data,axis=axis)
        var = xp.var(x.data,axis=axis)
        std = xp.sqrt(var + eps)

        if running_mean is not None and running_var is not None:
            N = np.prod(shape)
            sample_var = xp.multiply(var,N/(N-1),dtype=x.dtype)
            running_mean.data = (1-momentum) * running_mean.data + momentum*mean
            running_var.data = (1-momentum)*running_var.data + momentum*sample_var
    else:
        if running_mean is not None and running_var is not None:
            mean = running_mean.data
            var = running_var.data
            std = xp.sqrt(var + eps)
        else:
            mean = xp.mean(x.data,axis=axis)
            var = xp.var(x.data,axis=axis)
            std = xp.sqrt(var + eps)

    requires_grad = x.requires_grad

    mean,var,std = expand_dim(mean,var,std)
    norm = (x.data - mean)/std
    value = norm
    w_data = None

    if weight is not None:
        w_data = expand_dim(weight.data)[0]
        value *= w_data
        requires_grad |= weight.requires_grad
    if bias is not None:
        b_data = expand_dim(bias.data)[0]
        value += b_data
        requires_grad |= bias.requires_grad

    return Tensor(value,requires_grad,bias=bias,axis=axis,shape=shape,mean=mean,var=var+eps,std=std,norm=norm,weight_data=w_data)


def BatchNormBackward(x,w,child):
    
    axis= child.kwargs['axis']
    shape = child.kwargs['shape']
    norm = child.kwargs['norm']
    mean = child.kwargs['mean']
    var= child.kwargs['var']
    std = child.kwargs['std']
    bias = child.kwargs['bias']

    if w is not None:
        if w.requires_grad:
            w.grad += (norm*child.grad).sum(axis)

    if bias is not None:
        if bias.requires_grad:
            bias.grad += child.grad.sum(axis)

    if x.requires_grad:
        grad = child.grad
        if w is not None:
            grad = grad*child.kwrags['weight_data']
        N = np.prod(shape)
        a = xp.multiply(N,grad,dtype=grad.dtype) - grad.sum(axis=axis,keepdims=True)
        b = (x.data - mean) / var * xp.sum(grad * (x.data - mean),axis=axis,keepdims=True)
        value = xp.divide(a-b,N,dtype=grad.dtype) / std
        x.grad += value





















