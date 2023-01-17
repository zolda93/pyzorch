from zorch import*

def _im2col_gpu(img, kernel_size, stride, pad):
    """im2col function for GPU.
    This code is ported from Chainer:
    https://github.com/chainer/chainer/blob/v6.4.0/chainer/utils/conv.py
    """
    n, c, h, w = img.shape
    kh, kw = kernel_size
    sy, sx = stride
    ph, pw = pad
    out_h = (h + 2*ph - kh)//sy + 1
    out_w = (w + 2*pw - kw)//sx + 1
    dy, dx = 1, 1
    col = cp.empty((n, c, kh, kw, out_h, out_w), dtype=img.dtype)

    cp.ElementwiseKernel(
        'raw T img, int32 h, int32 w, int32 out_h, int32 out_w,'
        'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'
        'int32 dy, int32 dx',
        'T col',
        '''
           int c0 = i / (kh * kw * out_h * out_w);
           int ky = i / (kw * out_h * out_w) % kh;
           int kx = i / (out_h * out_w) % kw;
           int out_y = i / out_w % out_h;
           int out_x = i % out_w;
           int in_y = ky * dy + out_y * sy - ph;
           int in_x = kx * dx + out_x * sx - pw;
           if (in_y >= 0 && in_y < h && in_x >= 0 && in_x < w) {
             col = img[in_x + w * (in_y + h * c0)];
           } else {
             col = 0;
           }
        ''',
        'im2col')(img.reduced_view(),
                  h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dy, dx, col)


    return col

def _col2im_gpu(col, sy, sx, ph, pw, h, w):
    """col2im function for GPU.
    This code is ported from Chainer:
    https://github.com/chainer/chainer/blob/v6.4.0/chainer/utils/conv.py
    """
    n, c, kh, kw, out_h, out_w = col.shape
    dx, dy = 1, 1
    img = cp.empty((n, c, h, w),dtype=col.dtype)


    cp.ElementwiseKernel(
        'raw T col, int32 h, int32 w, int32 out_h, int32 out_w,'
        'int32 kh, int32 kw, int32 sy, int32 sx, int32 ph, int32 pw,'
        'int32 dx, int32 dy',
        'T img',
        '''
           int c0 = i / (h * w);
           int y  = i / w % h;
           int x  = i % w;
           T val = 0;
           for (int ky = 0; ky < kh; ++ky) {
             int out_y = (y + ph - ky * dy);
             if (0 > out_y || out_y >= out_h * sy) continue;
             if (out_y % sy != 0) continue;
             out_y /= sy;
             for (int kx = 0; kx < kw; ++kx) {
               int out_x = (x + pw - kx * dx);
               if (0 > out_x || out_x >= out_w * sx) continue;
               if (out_x % sx != 0) continue;
               out_x /= sx;
               int k = out_y + out_h * (kx + kw * (ky + kh * c0));
               val = val + col[out_x + out_w * k];
             }
           }
           img = val;
        ''',
        'col2im')(col.reduced_view(),
                  h, w, out_h, out_w, kh, kw, sy, sx, ph, pw, dx, dy, img)
    return img


def im2col(x,kernel_size,stride,padding):
    N,C,H,W = x.shape
    kh,kw = kernel_size
    sh,sw = stride
    ph,pw = padding
    OH = (H + 2*ph - kh)//sh + 1
    OW = (W + 2*pw - kw)//sw + 1
    

    if xp == cp:
        col = _im2col_gpu(x, kernel_size, stride, padding)
    else:
        x = np.pad(x,((0, 0), (0, 0), (ph, ph + sh - 1), (pw, pw + sw - 1)),mode='constant', constant_values=(0,))
        col = np.ndarray((N, C, kh, kw, OH, OW), dtype=img.dtype)

        for j in range(kh):
            j_lim = j + sh*OH
            for i in range(kw):
                i_lim = i + sw*OW
                col[:, :, j, i, :, :] = x[:, :, j:j_lim:sh, i:i_lim:sw]
    return col

def col2im(col,input_shape,kernel_size,stride,padding):
    N,C,H,W = input_shape
    kh,kw = kernel_size
    sh,sw = stride
    ph,pw = padding
    OH = (H + 2*ph - kh)//sh + 1
    OW = (W + 2*pw - kw)//sw + 1

    #col = col.reshape(N, OH, OW, C, kh, kw).transpose(0, 3, 4, 5, 1, 2)

    if xp == cp:
        x = _col2im_gpu(col, sh, sw, ph, pw, H, W)
    else:
        x = np.zeros((N, C, H + 2 * ph + sh - 1, W + 2 * pw + sw - 1),dtype=col.dtype)

        for j in range(kh):
            j_lim = j + sh*OH
            for i in range(kw):
                i_lim = i + sw*OW
                x[:, :, j:j_lim:sh, i:i_lim:sw] += col[:, :, j, i, :, :]
        x = x[:, :, ph:H + ph, Pw:W + pw]

    return x


def MaxPool2d(x,kernel_size,stride,padding,return_indices):

    x_data = x.data
    col = im2col(x_data,kernel_size,stride,padding)

    N,C,KH,KW,OH,OW = col.shape
    col = col.reshape(N,C,KH*KW,OH,OW)

    indexes = col.argmax(axis=2)

    value = col.max(axis=2)

    output = Tensor(value,x.requires_grad,kernel_size=kernel_size,stride=stride,padding=padding,indexes=indexes)
    return (output,indexes) if return_indices else output


def MaxPool2DWithIndicesBackward(x,child):

    if x.requires_grad:
        kernel_size = child.kwargs['kernel_size']
        stride = child.kwargs['stride']
        padding = child.kwargs['padding']
        indexes = child.kwargs['indexes']

        N,C,OH,OW = child.grad.shape
        N,C,H,W = x.shape
        KH,KW = kernel_size

        xgrad = xp.zeros((N*C*OH*OW*KH*KW))
        indexes = (indexes.ravel()+ xp.arange(0,indexes.size*KH*KW,KH*KW))

        xgrad[indexes] = child.grad.ravel()

        xgrad = xgrad.reshape(N, OH, OW, C, KH, KW).transpose(0, 3, 4, 5, 1, 2)

        xgrad = col2im(xgrad,(N,C,H,W),kernel_size,stride,padding)

        x.grad += xgrad


def AvgPool2d(x,kernel_size,stride,padding):

    value = im2col(x.data,kernel_size,stride,padding)
    value = value.mean(axis=(2,3))
    return Tensor(value,x.requires_grad,kernel_size=kernel_size,stride=stride,padding=padding)


def AvgPool2dBackward(x,child):

    if x.requires_grad:

        kh,kw = child.kwargs['kernel_size']
        stride = child.kwargs['stride']
        padding = child.kwargs['padding']

        N,C,OH,OW = child.grad.shape
        grad =xp.divide(child.grad,kh*kw,dtype=xp.float32)
        grad = xp.broadcast_to(grad.reshape(-1),(kh,kw,N*C*OH*OW))

        grad = grad.reshape(kh,kw,N,C,OH,OW).transpose(2,3,0,1,4,5)

        x.grad += col2im(grad,x.shape,(kh,kw),stride,padding)
