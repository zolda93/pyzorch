from zorch import*
from .optimizer import *

class SGD(Optimizer):
    def __init__(self,params,lr=required,momentum=0,dampening=0,
            weight_decay=0,nestrov=False):
        if lr is not required and lr <0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momntum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".fomat(weight_decay))

        defaults = dict(lr=lr,momentum=momentum,dampening=dampening,weight_decay=weight_decay,nestrov=nestrov)
        if nestrov and (momentum < 0.0 or dampening != 0):
            raise ValueError("Nestrov momentum requires a momentum and zero dampening")
        super().__init__(params,defaults)

    def __setstate__(self,state):
        super(SGD,self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nestrov',False)

    def step(self):
        for group in self.param_groups:
            params_with_grad = []
            d_p_list = []
            momentum_buffer_list = []
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            lr = group['lr']
            nestrov = group['nestrov']

            for p in group['params']:
                if p.grad.__class__ is not int:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(None)
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

            sgd(params_with_grad,d_p_list,momentum_buffer_list,weight_decay,momentum,lr,dampening,nestrov)
            
            for p,momentum_buffer in zip(params_with_grad,momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer


def sgd(params,d_p_list,momentum_buffer_list,weight_decay,momentum,lr,dampening,nestrov):
    for i,p in enumerate(params):
        d_p = d_p_list[i]

        if weight_decay != 0:
            d_p = d_p + p.data*weight_decay

        if momentum != 0:
            buf = momentum_buffer_list[i]
            if buf is None:
                buf = d_p
                momentum_buffer[i] = buf
            else:
                buf *= momentum
                buf += d_p*(1-dampening)
            if nestrov:
                d_p = d_p + buf*momentum
            else:
                d_p = buf
        p.data -= d_p*lr



        

