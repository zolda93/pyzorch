import math
from zorch import zeros ,xp
from .module import *
from .. import functional as F
from .. import init

class ConvNd(Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,
            padding,dilation,transposed,output_padding,groups,bias,padding_mode='zeros'):
        super().__init__()

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride = stride
        self.padding = padding 
        self.dilation = dilation
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        self.transposed = transposed

        if self.transposed:
            self.weight = Parameter(zeros((in_channels,out_channels//groups,*kernel_size)))
        else:
            self.weight = Parameter(zeros((out_channels,in_channels//groups,*kernel_size)))

        if bias:
            self.bias = Parameter(zeros(out_channels))
        else:
            self.register_parameter('bias',None)

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform(self.weight,a=math.sqrt(5))
        if self.bias is not None:
            fan_in,_ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1/math.sqrt(fan_in)
            self.bias.data = xp.random.uniform(low=-bound, high=bound, size=self.bias.shape).astype(xp.float32)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)


class Conv2d(ConvNd):
    def __init__(self,in_channels,out_channels,kernel_size,
            stride=1,padding=0,dilation=1,groups=1,bias=True,padding_mode='zeros'):
        kernel_size = F.pair(kernel_size)
        stride = F.pair(stride)
        padding = F.pair(padding)
        dilation = F.pair(dilation)

        super().__init__(in_channels,out_channels,kernel_size,stride,padding,dilation,False,F.pair(0),groups,bias,padding_mode)

    
    def forward(self,x):

        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(x,self.padding,self.padding_mode),self.weight,self.bias,self.stride,F.pair(0),self.dilation,self.groups)

        return F.conv2d(x,self.weight,self.bias,self.stride,self.padding,self.dilation,self.groups)




class ConvTransposeNd(ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding,
                 groups, bias, padding_mode):
        if padding_mode != 'zeros':
            raise ValueError('Only "zeros" padding mode is supported for {}'.format(self.__class__.__name__))

        super().__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, transposed, output_padding,
            groups, bias, padding_mode)

    def determing_output_padding(self, input, output_size, stride, padding, kernel_size, dilation=None):
        if output_size is None:
            ret = F.single(self.output_padding)  # converting to list if was not already
        else:
            k = input.dim() - 2
            if len(output_size) == k + 2:
                output_size = output_size[2:]
            if len(output_size) != k:
                raise ValueError(
                    "output_size must have {} or {} elements (got {})"
                        .format(k, k + 2, len(output_size)))

            min_sizes = []
            max_sizes = []
            for d in range(k):
                dim_size = ((input.shape[d + 2] - 1) * stride[d] -
                            2 * padding[d] +
                            (dilation[d] if dilation is not None else 1) * (kernel_size[d] - 1) + 1)
                min_sizes.append(dim_size)
                max_sizes.append(min_sizes[d] + stride[d] - 1)

            for i in range(len(output_size)):
                size = output_size[i]
                min_size = min_sizes[i]
                max_size = max_sizes[i]
                if size < min_size or size > max_size:
                    raise ValueError((
                        "requested an output size of {}, but valid sizes range "
                        "from {} to {} (for an input of {})").format(
                        output_size, min_sizes, max_sizes, input.shape[2:]))

            res = []
            for d in range(k):
                res.append(output_size[d] - min_sizes[d])

            ret = res
        return ret


class ConvTranspose2d(ConvTransposeNd):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            output_padding=0,
            groups=1,
            bias=True,
            dilation=1,
            padding_mode='zeros'
    ):
        kernel_size = F.pair(kernel_size)
        stride = F.pair(stride)
        padding = F.pair(padding)
        dilation = F.pair(dilation)
        output_padding = F.pair(output_padding)
        super().__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            True, output_padding, groups, bias, padding_mode)

    def forward(self, x, output_size=None):
        if self.padding_mode != 'zeros':
            raise ValueError('Only `zeros` padding mode is supported for ConvTranspose2d')

        output_padding = self.determing_output_padding(
            x, output_size, self.stride, self.padding, self.kernel_size, self.dilation)

        return F.conv_transpose2d(
            x, self.weight, self.bias, self.stride, self.padding,
            output_padding, self.groups, self.dilation)
