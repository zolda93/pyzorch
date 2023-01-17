import functools

grad_enable = True

def is_grad_enable()->bool:
    return grad_enable

def set_grad_enable(mode:bool)->None:
    global grad_enable
    grad_enable = mode

class no_grad:

    def __enter__(self)->None:
        self.prev = is_grad_enable()
        set_grad_enable(False)

    def __exit__(self,exc_type,exc_value,trace)->None:
        set_grad_enable(self.prev)

    def __call__(self,func):
        @functools.wraps(func)
        def decorate_context(*args,**kwargs):
            with __class__():
                return func(*args,**kwargs)
        return decorate_context


class enable_grad:
    def __enter__(self)->None:
        self.prev = is_grad_enable()
        set_grad_enable(True)

    def __exit__(self,exc_type,exc_value,trace)->None:
        set_grad_enable(self.prev)
    
    def __call__(self, func):
        @functools.wraps(func)
        def decorate_context(*args,**kwargs):
            with __class__():
                return func(*args,**kwargs)
        return decorate_context


