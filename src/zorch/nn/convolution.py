from zorch import*
import math

import cupy.fft as fft

rfft2 = fft.rfft2
irfft2 = fft.irfft2

def as_strided(x,shape,strides):
    return xp.lib.stride_tricks.as_strided(x,shape=shape,strides=strides)


def determing_pad_shape(x,w,padding):
    return (x.shape[-2] + w.shape[-2] - 1 + padding[0]*2,x.shape[-1] + w.shape[-1] - 1 + padding[1]*2)

def pad_right(x,shape,padding):
    hp,wp = padding
    ho,wo = shape
    pad_with = [[0,0] for _ in range(x.ndim)]
    pad_with[-2] = [hp,ho - x.shape[-2] - hp]
    pad_with[-1] = [wp,wo - x.shape[-1] - wp]
    return xp.pad(x,pad_with)

def _repeat(x,axis,reps):
    shape = *x.shape[:axis],reps,*x.shape[axis:]
    strides = *x.strides[:axis],0,*x.strides[axis:]
    return as_strided(x,shape,strides)

def _split(x,axis,groups):

    group_size = x.shape[axis] // groups
    shape = *x.shape[:axis],groups,group_size,*x.shape[(axis+1):]
    strides = *x.strides[:axis],group_size * x.strides[axis],x.strides[axis],*x.strides[(axis+1):]
    return as_strided(x,shape,strides),group_size


def dilate(x,dilation):
    hd,wd = dilation
    h,w = x.shape[-2:]
    dilated = xp.zeros((*x.shape[:-2], (h - 1) * hd + 1, (w - 1) * wd + 1))
    dilated[...,::hd,::wd] = x
    return dilated


def _fft(padding,output_shape,groups,x=None,w=None,grad=None,kind="x"):

    if kind == "x":
        x_padded = pad_right(x,output_shape,padding)
        w_padded = pad_right(xp.flip(w,axis=(-1,-2)),output_shape,(0,0))

        x_fft = rfft2(x_padded)
        w_fft = rfft2(w_padded)

        w_fft = _split(w_fft,-4,groups)[0]
        x_fft = _split(x_fft,-3,groups)[0]
        x_fft = _repeat(x_fft,-3,groups)
        return x_fft,w_fft

    elif kind == "dx":
        w_padded = pad_right(w,output_shape,(0,0))
        g_padded = pad_right(grad,output_shape,padding)

        w_fft = rfft2(w_padded)
        g_fft = rfft2(g_padded)

        w_fft,w_groups_size = _split(w_fft,-4,groups)
        x_group_size = w.shape[-3]
        g_fft = _repeat(_split(g_fft,-3,groups)[0],-2,x_group_size)

        return w_fft,g_fft
    
    elif kind == "dw":
        x_padded = pad_right(x,output_shape,padding)
        g_padded = pad_right(xp.flip(grad,axis=(-1,-2)),output_shape,(0,0))

        x_fft = rfft2(x_padded)
        g_fft = rfft2(g_padded)

        x_fft,x_group_size = _split(x_fft,-3,groups)
        w_group_size = w.shape[-4] // groups
        x_fft = _repeat(x_fft,-3,w_group_size)
        g_fft = _repeat(_split(g_fft,-3,groups)[0],-2,x_group_size)
        return x_fft,g_fft


def _ifft(value,output_shape,x_shape=None,w_shape=None,grad_shape=None,padding=None,kind="x"):

    if kind == "x":
        hp,wp = padding
        hi,wi = x_shape[-2:]
        hk,wk = w_shape[-2:]
        value = irfft2(value,s=output_shape)[...,(hk - 1):(hi + 2 * hp), (wk - 1):(wi + 2 * wp)]
        return value

    elif kind == "dx":
        hg,wg = grad_shape[-2:]
        hk,wk = w_shape[-2:]
        hp,wp = padding
        
        value = irfft2(value,s=output_shape)[...,:(hg + hk + hp - 1), :(wg + wk + wp - 1)]
        return value
    elif kind == "dw":
        hp,wp = padding
        hi,wi = x_shape[-2:]
        hk,wk = grad_shape[-2:]
        value = irfft2(value,s=output_shape)[..., (hk - 1):(hi + 2 * hp),(wk - 1):(wi + 2 * wp)]
        return value


def _conv2d(x,w,stride,padding,dilation,groups,weight_dilated=None):

    if weight_dilated is None:
        weight_dilated = dilate(w,dilation)

    output_shape = determing_pad_shape(x,weight_dilated,padding)

    x_fft,w_fft = _fft(padding,output_shape,groups,x=x,w=w)

    value = xp.squeeze(xp.moveaxis(x_fft[...,None],-4,-1) @ xp.moveaxis(w_fft,-3,-1)[...,None],(-1,-2))
    value = xp.reshape(value,(*value.shape[:-4],-1,*value.shape[-2:]))

    value = _ifft(value,output_shape,x.shape,weight_dilated.shape,grad_shape=None,padding=padding)

    value = value[...,::stride[0],::stride[1]]
    return value,weight_dilated

def _dx(grad,w,stride,padding,dilation,groups,grad_dilated=None,weight_dilated=None,x=None,output_padding=(0,0)):

    if grad_dilated is None:
        grad_dilated = dilate(grad,stride)

    if weight_dilated is None:
        weight_dilated = dilate(w,dilation)

    output_shape = determing_pad_shape(grad_dilated,weight_dilated,padding)

    w_fft,g_fft = _fft(padding,output_shape,groups,x=None,w=weight_dilated,grad=grad_dilated,kind="dx")

    value = xp.squeeze(xp.moveaxis(g_fft[..., None], -5, -1) @ xp.moveaxis(w_fft, -4, -1)[..., None], (-1, -2))
    value = xp.reshape(value,(*value.shape[:-4], -1, *value.shape[-2:]))
    
    if x is not None:
        hi,wi = x.shape[-2:]
    else:
        hop = output_padding[0]
        wop = hop if len(output_padding) == 1 else output_padding[1]
        hi = (grad.shape[-2] - 1) * stride[0] - 2 * padding[0] + dilation[0] * (w.shape[-2] - 1) + hop  + 1
        wi = (grad.shape[-1] - 1) * stride[1] - 2 * padding[1] + dilation[1] * (w.shape[-1] - 1) + wop  + 1

    value = _ifft(value,output_shape,x_shape=None,w_shape=weight_dilated.shape,grad_shape=grad_dilated.shape,padding=padding,kind="dx")
    value = value[...,2 * padding[0]:(2 * padding[0] + hi), 2 * padding[1]:(2 * padding[1] + wi)]

    h,w = value.shape[-2:]
    value = xp.pad(value,((0, 0), (0, 0), (0, hi - h), (0, wi - w)))

    return value,weight_dilated,grad_dilated

def _dw(x,grad,w,stride,padding,dilation,groups,grad_dilated=None):

    if grad_dilated is None:
        grad_dilated = dilate(grad,stride)

    output_shape = determing_pad_shape(x,grad_dilated,padding)

    x_fft,g_fft = _fft(padding,output_shape,groups,x=x,w=w,grad=grad_dilated,kind="dw")

    value = xp.squeeze(xp.moveaxis(x_fft[..., None], 0, -1) @ xp.moveaxis(g_fft, 0, -1)[..., None],(-1, -2))
    value= xp.reshape(value,(*value.shape[:-5], -1, *value.shape[-3:]))

    value = _ifft(value,output_shape,x_shape=x.shape,w_shape=None,grad_shape=grad_dilated.shape,padding=padding,kind="dw")
    hs, ws = dilation
    hk, wk = w.shape[-2:]
    value = value[..., ::hs, ::ws]
    value = value[..., :hk, :wk]

    return value,grad_dilated



def Conv2d(x,w,bias,stride,padding,dilation,groups):

    if x.ndim < 4:
        raise RuntimeError(f"Expected 4D (batched) input to conv2d,but got input of size: {x.data.shape}")

    if groups * w.shape[-3] != x.shape[-3]:
        raise RuntimeError(f"'Given groups={groups}, weight of size {w.shape}, expected input{x.shape} to have {groups*w.shape[-3]} channels, but got {x.shape[-3]} channels instead")

    x = x.contiguous()

    requires_grad = x.requires_grad | w.requires_grad

    value,weight_dilated = _conv2d(x.data,w.data,stride,padding,dilation,groups)

    if bias is not None:
        requires_grad |= bias.requires_grad

        value += bias.data[:,None,None]

    return Tensor(value,requires_grad=requires_grad,bias=bias,stride=stride,padding=padding,dilation=dilation,groups=groups,weight_dilated=weight_dilated,transpose=False)


def ConvTranspose2d(x,w,bias,stride,padding,output_padding,groups,dilation):

    if x.ndim < 4:
        raise RuntimeError(f"'Expected 4D (batched) input to TransposeConv2d,but got input of size: {x.shape}")
    if w.shape[-4] != x.shape[-3]:
        raise RuntimeError(f"Given transposed=1, weight of size {w.shape}, expected input {x.shape} to have {w.shape[-4]} channels, but got {x.shape[-3]} channels instead")


    x = x.contiguous()
    requires_grad = x.requires_grad | w.requires_grad

    value,weight_dilated,grad_dilated = _dx(x.data,w.data,stride,padding,dilation,groups,output_padding=output_padding)

    if bias is not None:
        requires_grad |= bias.requires_grad

        value += bias.data[:,None,None]

    return Tensor(value,requires_grad=requires_grad,bias=bias,stride=stride,padding=padding,dilation=dilation,groups=groups,weight_dilated=weight_dilated,grad_dilated=grad_dilated,transpose=True)


def ConvolutionBackward(x,w,child):

    bias = child.kwargs['bias']
    stride = child.kwargs['stride']
    padding = child.kwargs['padding']
    dilation = child.kwargs['dilation']
    groups = child.kwargs['groups']
    transpose = child.kwargs['transpose']

    if transpose == False:
        if bias is not None:
            if bias.requires_grad:
                bias.grad += xp.einsum('nohw->o',child.grad)

        if w.requires_grad:
            wgrad,grad_dilated = _dw(x.data,child.grad,w.data,stride,padding,dilation,groups)
            w.grad += wgrad

        if x.requires_grad:
            weight_dilated = child.kwargs['weight_dilated']
            xgrad = _dx(child.grad,w.data,stride,padding,dilation,groups,grad_dilated,weight_dilated,x.data)[0]
            x.grad += xgrad
            x.grad[xp.abs(x.grad) < 1e-5] = 0
    else:
        if bias is not None:
            if bias.requires_grad:
                bias.grad += xp.einsum('Nohw->o',child.grad)
        if w.requires_grad:
            grad_dilated = child.kwargs['grad_dilated']
            w.grad += _dw(child.grad, x.data, w.data, stride, padding,dilation, groups, grad_dilated)[0]
        if x.requires_grad:
            weight_dilated = child.kwargs['weight_dilated']
            x.grad += _conv2d(child.grad,w.data,stride,padding,dilation,groups,weight_dilated)[0]
            x.grad[xp.abs(x.grad) < 1e-5] = 0





















