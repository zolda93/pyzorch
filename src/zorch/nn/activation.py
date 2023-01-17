from zorch import*


def Tanh(x):
    return Tensor(xp.tanh(x.data),x.requires_grad)

def TanhBackward(x,child):
    if x.requires_grad:
        x.grad += child.grad * (1 - child.data*child.data)

def ReLU(x,inplace):
    if inplace:
        x.data = x.data > 0
        value = x.data
    else:
        value = x.data > 0

    return Tensor(value,x.requires_grad)

def ReLUBackward(x,child):
    if x.requires_grad:
        x.grad += child.grad*(x.data > 0)

def ReLU6(x,inplace):
    if inplace:
        x.data = xp.minimum(xp.maximum(x.data,0),6)
        value = x.data
    else:
        value = xp.minimum(xp.maximum(x.data,0),6)

    return Tensor(value,x.requires_grad)

def ReLU6Backward(x,child):
    if x.requires_grad:
        xgrad = child.grad
        xgrad[x.data < 0] = 0
        xgrad[x.data > 6] = 0
        x.grad += xgrad


def LeakyReLU(x,negative_slope,inplace):
    value = xp.maximum(x.data*negative_slope,x.data,out=x.data if inplace else None)
    return Tensor(value,x.requires_grad,negative_slope=negative_slope)

def LeakyReLUBackward(x,child):
    if x.requires_grad:
        negative_slope = child.kwargs['negative_slope']

        value = xp.ones_like(x.data)
        value[x.data < 0] = negative_slope
        x.grad += value*child.grad

def Sigmoid(x):
    value = xp.divide(1,xp.add(1,xp.exp(-x.data),dtype=x.dtype),dtype=x.dtype)
    return Tensor(value,x.requires_grad)

def SigmoidBackward(x,child):
    if x.requires_grad:
        x.grad += child.grad * child.data * (1 - child.data)

def GLU(x,dim):
    a ,b = xp.split(x.data,2,axis=dim)
    b_sigmoid = xp.divide(1,xp.add(1,xp.exp(-b),dtype=x.dtype),dtype=x.dtype)
    value = a * b_sigmoid
    return Tensor(value,x.requires_grad,a=a,b=b_sigmoid,dim=dim)

def GLUBackward(x,child):
    if x.requires_grad:
        dim = child.kwargs['dim']
        a = child.kwargs['a']
        b_sigmoid = child.kwargs['b']

        agrad = child.grad * b_sigmoid
        bgrad = child.grad * a * b_sigmoid*(1-b_sigmoid)
        x.grad += xp.concatenate((agrad,bgrad),axis=dim)

def ELU(x,alpha,inplace):
    if inplace:
        x.data = xp.where(x.data > 0,x.data,alpha*(xp.exp(x.data)-1))
        value = x.data
    else:
        value = xp.where(x.data > 0,x.data,alpha*(xp.exp(x.data)-1))
    return Tensor(value,x.requires_grad,alpha=alpha)

def ELUBackward(x,child):
    if x.requires_grad:
        alpha = child.kwargs['alpha']
        xgrad = child.grad * xp.where(x.data > 0,xp.ones_like(x.data),alpha*xp.exp(x.data))
        x.grad += xgrad

def SELU(x,inplace):

    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    if inplace:
        x.data = xp.where(x.data > 0,x.data,alpha*(xp.exp(x.data)-1))
        value = x.data
    else:
        value = xp.where(x.data > 0,x.data,alpha*(xp.exp(x.data)-1))
    value *= scale
    return Tensor(value,x.requires_grad,alpha=alpha,scale=scale)

def SELUBackward(x,child):
    if x.requires_grad:
        alpha = child.kwargs['alpha']
        scale = child.kwargs['scale']
        x.grad += child.grad * xp.where(x.data >= 0,xp.ones_like(x.data)*scale,xp.exp(x.data) * alpha*scale)

def Softmax(x,axis):
    aug = x.data - xp.max(x.data,axis=axis,keepdims=True)
    exp = xp.exp(aug)
    sum_exp = xp.sum(exp,axis=axis,keepdims=True)
    return Tensor(exp/sum_exp,x.requires_grad,axis=axis)

def SoftmaxBackward(x,child):
    if x.requires_grad:
        axis = child.kwargs['axis']
        x.grad += (child.grad - xp.sum((child.grad * child.data),axis=axis,keepdims=True))*child.grad



def LogSigmoid(x):
    value = x.data * (x.data < 0) -xp.log1p(xp.exp(-xp.abs(x.data)))
    return Tensor(value,x.requires_grad)

def LogSigmoidBackward(x,child):
    if x.requires_grad:
        x.grad += child.grad * xp.exp(-x.data + child.data)


def LogSoftmax(x,axis=None):

    aug = x.data - xp.max(x.data,axis=axis,keepdims=True)
    exp = xp.exp(aug)
    sum_exp = xp.sum(exp,axis=axis,keepdims=True)
    log_sum_exp = xp.log(sum_exp)

    return Tensor(aug - log_sum_exp,x.requires_grad,axis=axis,softmax=exp/sum_exp)


def LogSoftmaxBackward(x,child):

    if x.requires_grad:

        axis=child.kwargs['axis']
        softmax=child.kwargs['softmax']

        x.grad += child.grad - xp.sum(child.grad,axis=axis,keepdims=True)*softmax
