from zorch.op import UnaryOp,BinaryOp
from .loss import*
from .activation import *
from .linear import*
from .flatten import*
from .dropout import*
from .pooling import*
from .padding import*
from .convolution import*
from .normalization import*


class _l1loss(BinaryOp):
    func = L1Loss
    grad_func = L1LossBackward

class _mseloss(BinaryOp):
    func = MSELoss
    grad_func = MSELossBackward

class _bceloss(BinaryOp):
    func = BCELoss
    grad_func = BCELossBackward

class _bcewithlogitsloss(BinaryOp):
    func = BCEWithLogitsLoss
    grad_func = BCEWithLogitsLossBackward

class _nllloss(BinaryOp):
    func = NllLoss
    grad_func = NllLossBackward

class _relu(UnaryOp):
    func = ReLU
    grad_func = ReLUBackward

class _leakyrelu(UnaryOp):
    func = LeakyReLU
    grad_func = LeakyReLUBackward

class _tanh(UnaryOp):
    func = Tanh
    grad_func = TanhBackward

class _relu6(UnaryOp):
    func = ReLU6
    grad_func = ReLU6Backward

class _glu(UnaryOp):
    func = GLU
    grad_func = GLUBackward

class _elu(UnaryOp):
    func = ELU
    grad_func = ELUBackward

class _selu(UnaryOp):
    func = SELU
    grad_func = SELUBackward

class _sigmoid(UnaryOp):
    func = Sigmoid
    grad_func = SigmoidBackward

class _logsigmoid(UnaryOp):
    func = LogSigmoid
    grad_func = LogSigmoidBackward

class _softmax(UnaryOp):
    func = Softmax
    grad_func = SoftmaxBackward

class _logsoftmax(UnaryOp):
    func = LogSoftmax
    grad_func = LogSoftmaxBackward

class _linear(BinaryOp):
    func = Linear
    grad_func = LinearBackward

class _flatten(UnaryOp):
    func = Flatten
    grad_func = FlattenBackward

class _dropout(UnaryOp):
    func = Dropout
    grad_func = DropoutBackward

class _maxpool2d(UnaryOp):
    func = MaxPool2d
    grad_func = MaxPool2DWithIndicesBackward

class _avgpool2d(UnaryOp):
    func = AvgPool2d
    grad_func = AvgPool2dBackward
class _pad(UnaryOp):
    func = Pad
    grad_func = PadBackward

class _conv2d(BinaryOp):
    func = Conv2d
    grad_func = ConvolutionBackward

class _convtranspose2d(BinaryOp):
    func = ConvTranspose2d
    grad_func = ConvolutionBackward

class _layernorm(BinaryOp):
    func = LayerNorm
    grad_func = LayerNormBackward

class _batchnorm(BinaryOp):
    func = BatchNorm
    grad_func = BatchNormBackward
