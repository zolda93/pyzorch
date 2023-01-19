# pyzorch
a deep learning framework style pytorch,implemented entirely in numpy, with GPU acceleration from cupy.
the goal was to get a deep undertanding of how framework like pytorch work under the hood

## Quick start
```
import zorch
from zorch.nn import functional as F
a = zorch.tensor([[2., 2., 1.],[0., 0., 1.],[3., 2., 0.]],requires_grad=True)
b = zorch.tensor([[0., 3., 2.],[1., 2., 0.],[1., 1., 4.]],requires_grad=True)
print(a)
print(b)
tensor([[2., 2., 1.],
        [0., 0., 1.],
        [3., 2., 0.]], requires_grad=True)
tensor([[0., 3., 2.],
        [1., 2., 0.],
        [1., 1., 4.]], requires_grad=True)


x = a+b
y = a-b
z = x@y.transpose()
t = F.elu(z)
w = t.sum()
print(x)
print(y)
print(z)
print(t)
print(w)
w.backward()

tensor([[2., 5., 3.],
        [1., 2., 1.],
        [4., 3., 4.]],grad_fn=<AddBackward0>)
tensor([[ 2., -1., -1.],
        [-1., -2.,  1.],
        [ 2.,  1., -4.]],grad_fn=<SubBackward0>)
tensor([[-4., -9., -3.],
        [-1., -4.,  0.],
        [ 1., -6., -5.]],grad_fn=<MmBackward0>)
tensor([[-0.9817, -0.9999, -0.9502],
        [-0.6321, -0.9817,  0.    ],
        [ 1.    , -0.9975, -0.9933]],grad_fn=<ELUBackward0>)
tensor(-5.5364,grad_fn=<SumBackward0>)

print(w.grad)
print(t.grad)
print(z.grad)
print(x.grad)
print(y.grad)
print(a.grad)
print(b.grad)

1.0
[[1. 1. 1.]
 [1. 1. 1.]
 [1. 1. 1.]]
[[1.8316e-02 1.2341e-04 4.9787e-02]
 [3.6788e-01 1.8316e-02 1.0000e+00]
 [1.0000e+00 2.4788e-03 6.7379e-03]]
[[ 0.1361  0.0312 -0.2173]
 [ 2.7174  0.5955 -4.3496]
 [ 2.011  -0.9982 -1.0245]]
[[4.4045 3.8273 4.4228]
 [0.0285 0.0447 0.0286]
 [1.1265 2.2691 1.1763]]
[[ 4.5406  3.8586  4.2055]
 [ 2.7459  0.6402 -4.321 ]
 [ 3.1375  1.2709  0.1518]]
[[-4.2684 -3.7961 -4.6402]
 [ 2.689   0.5508 -4.3782]
 [ 0.8845 -3.2674 -2.2008]]
 
 ```
 
 ## Supported functionalities
### Misc. 
#### variable functions
`ones`, `ones_like`, `zeros`, `zeros_like`, `randn`, `tensor`, ...   
 
### Autograd, forward and backward propagations of:
#### basic operations   
add: `+`, sub: `-`, mul: `*`, div: `/`, `sum`, `mean`, `var`      

#### ufunc  
`exp`, `log`, `sin`, `cos`, `tanh`, `sigmoid`, power: `x**2`, exponential: `2**x`,

#### tensor  
matmul: `@`, elementwise (Hadamard) mul: `*`, transpose: `.T` or `transpose(x, axes)`, concatenate: `concat`, view: `view`, slicing: `x[...,1:10:2,:]`, `flatten`, etc.       

#### nn.modules  
* activation: `Tanh`, `Sigmoid`, `LogSigmoid`, `ReLU`, `LeakyReLU`, `Softmax`, `LogSoftmax`, `Elu`,`Selu`,`Glu`    
* batchnorm: `BatchNorm1d`, `BatchNorm2d`, `BatchNorm3d`   
* container: `Sequential`, `ModuleList`  
* conv: `Conv2d`  
* transposed conv: `ConvTranspose2d`
* dropout: `Dropout`    
* linear: `Linear`    
* loss: `L1Loss`,`MSELoss`, `BCELoss`, `BCEWithLogitsLoss`, `NLLLoss`,`CrossEntropyLoss`  
* module: `Module`  
* pooling: `MaxPool2d`, `AvgPool2d`  
 
 

#### optim    
adapted from pytorch code.
* optimizer: `Optimizer`   
* sgd: `SGD`
* Adam: `Adam`  
* Adamw: `AdamW`  
* Adadelta:`Adadelta`

### zorch.utils.data

 * Dataset : `Dataset`
 * Dataloader : `DataLoader`

### zorchvision

* transforms : `Compose`,`ToTensor`,`Resize`,`Normalize`,`CenterCrop`,`RandomHorizontalFlip`,`RandomVerticalFlip`
* datasets   : `MNIST`,`FashionMNIST`,`Cifar10`,`Cifar100`,`STL10`

### Resourses
 
[chainer](https://github.com/chainer/chainer)

[pytorch](https://github.com/pytorch/pytorch)

[tinygrad](https://github.com/geohot/tinygrad)

[ToeffiPy](https://github.com/ChristophReich1996/ToeffiPy/tree/master/autograd)


