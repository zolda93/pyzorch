from zorch import*
from .module import Module
from .. import functional as F
from .. import  init
from ..parameter import Parameter



class NormBase(Module):
    def __init__(self,num_features,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True):
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        if self.affine:
            self.weight = Parameter(ones((self.num_features)))
            self.bias = Parameter(zeros((self.num_features)))
        else:
            self.register_parameter('weight',None)
            self.register_parameter('bias',None)

        if self.track_running_stats:
            self.register_buffer('running_mean',zeros((num_features)))
            self.register_buffer('running_var',ones((num_features)))
            self.register_buffer('num_batches_tracked',Tensor(0))
        else:
            self.register_parameter('running_mean',None)
            self.register_parameter('running_var',None)
            self.register_parameter('num_batches_tracked',None)
        
        self.reset_parameters()


    def reset_running_stats(self):

        if self.track_running_stats:
            self.running_mean = init.zeros(self.running_mean)
            self.running_var = init.ones(self.running_var)
            self.num_batches_tracked = init.zeros(self.num_batches_tracked)


    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.ones(self.weight)
            init.zeros(self.bias)

    def _check_input_dim(self,x):
        raise NotImplementedError

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine},track_running_stats={track_running_stats}'.format(**self.__dict__)



class BatchNorm(NormBase):
    def __init__(self,num_features,eps=1e-5,momentum=0.1,affine=True,track_running_stats=True):
        super().__init__(num_features,eps,momentum,affine,track_running_stats)


    def forward(self,x):
        self._check_input_dim(x)
        if self.training and self.track_running_stats:
            if self.momentum is None:
                exp_avg_factor = 1.0 / self.num_batches_tracked.data
            else:
                exp_avg_factor = self.momentum
        else:
            exp_avg_factor = None


        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        return F.batch_norm(x,
                self.running_mean if not self.training or self.track_running_stats else None,
                self.running_var if not self.training or self.track_running_stats else None,
                self.weight,self.bias,bn_training,exp_avg_factor,self.eps)


class BatchNorm1d(BatchNorm):
    def _check_input_dim(self,x):
        if x.ndim != 2 and x.ndim != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'.format(x.ndim))


class BatchNorm2d(BatchNorm):
    def _check_input_dim(self,x):
        if x.ndim != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(x.ndim))

class BatchNorm3d(BatchNorm):
    def _check_input_dim(self,x):
        if x.ndim != 5:
            raise ValueError('expected 5D input (got {}D input)'.format(x.ndim))
