from zorch import*
from .optimizer import*

class Adadelta(Optimizer):
    def __init__(self,params,lr=1.0,rho=0.9,eps=1e-6,weight_decay=0):

        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= rho <= 1.0:
            raise ValueError("Invalid rho value: {}".format(rho))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def __setstate__(self,state):
        super().__setstate__(state)

    def step(self):
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            square_avgs = []
            acc_deltas = []
            lr = group['lr']
            rho = group['rho']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad.__class__ is int:
                    continue
                params_with_grad.append(p)
                grads.append(p.grad)
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = xp.zeros_like(p.data)
                    state['acc_delta'] = xp.zeros_like(p.data)
                square_avgs.append(state['square_avg'])
                acc_deltas.append(state['acc_delta'])
                state['step'] += 1

            adadelta(params_with_grad,
                    grads,
                    square_avgs,
                    acc_deltas,
                    lr=lr,
                    rho=rho,
                    eps=eps,
                    weight_decay=weight_decay)


def adadelta(params,grads,square_avgs,acc_deltas,lr,rgo,eps,weight_decay):

    for (param,grad,square_avg,acc_delta) in zip(params,grads,square_avgs,acc_deltas):

        if weight_decay != 0:
            grad += p.data*weight_decay


        square_avg = square_avg * rho + (1-rho)*grad*grad
        delta = xp.sqrt((acc_delta + eps)/(square_avg + eps))*grad
        acc_delta = acc_delta*rho + delta*delta*(1-rho)
        p.data -= delta*lr

