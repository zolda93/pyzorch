from zorch import*
from .optimizer import*

class AdamW(Optimizer):
    def __init__(self,params,lr=1e-3,betas=(0.0,0.999),eps=1e-8,weight_decay=1e-2,amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay,amsgrad=amsgrad)
        super().__init__(params,defaults)


    def __setstate__(self,state):
        super().__setstate__(state)
        for group in self.param_groups:
            groups.setdefault('amsgrad',False)

    def step(self):
        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group['amsgrad']
            beta1,beta2 = group['betas']

            for p in group['params']:
                if p.grad.__class__ is int:
                    continue
                params_with_grad.append(p)
                grads.append(p.grad)
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = xp.zeros_like(p.data)
                    state['exp_avg_sq' ] =xp.zeros_like(p.data)
                    if amsgrad:
                        state['max_exp_avg_sq'] = xp.zeros_like(p.data)
                exp_avgs.append(state['exp_avg'])
                exp_avg_sqs.append(state['exp_avg_sq'])
                if amsgrad:
                    max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                state['step'] += 1
                state_steps.append(state['step'])

            adamw(params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    state_steps,
                    amsgrad = amsgrad,
                    beta1 = beta1,
                    beta2=beta2,
                    lr = group['lr'],
                    weight_decay=group['weight_decay'],
                    eps=group['eps'])


def adamw(params,grads,exp_avgs,exp_avg_sqs,max_exp_avg_sqs,state_steps,amsgrad,beta1,beta2,lr,weight_decay,eps):
    for i,p in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        # Perform stepweight decay
        p.data *= 1 - lr * weight_decay

        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step

        # Decay the first and second moment running average coefficient
        exp_avg *= beta1
        exp_avg += (1 - beta1) * grad

        exp_avg_sq *= beta2
        exp_avg_sq += grad * grad.conj() * (1 - beta2)

        if amsgrad:
            # Maintains the maximum of all 2nd moment running avg. till now
            xp.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
            # Use the max. for normalizing running avg. of gradient
            denom = (xp.sqrt(max_exp_avg_sqs[i]) / xp.sqrt(bias_correction2)) + eps
        else:
            denom = (xp.sqrt(exp_avg_sq) / xp.sqrt(bias_correction2)) + eps

        step_size = lr / bias_correction1
        p.data += -step_size * exp_avg / denom


