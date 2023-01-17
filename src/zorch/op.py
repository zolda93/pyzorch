from abc import ABC,abstractmethod

class Op(ABC):
    func = None
    grad_func = None

    def __init__(self,parents=[]):
        self.parents = [p for p in parents]
        self.child = None

    @abstractmethod
    def forward(self,*args,**kwargs):
        raise NotImplementedError

    @abstractmethod
    def backward(self,child=None):
        raise NotImplementedError



class UnaryOp(Op):

    def __init__(self,parent):
        super().__init__([parent])

    def forward(self,*args,**kwargs):
        self.child = self.__class__.func(self.parents[0],*args,**kwargs)
        return self.child
    
    def backward(self,child):

        assert (child is not None)
        self.__class__.grad_func(self.parents[0],child)


class BinaryOp(Op):

    def __init__(self,parent0,parent1):
        super().__init__([parent0,parent1])

    def forward(self,*args,**kwargs):
        self.child = self.__class__.func(self.parents[0],self.parents[1],*args,**kwargs)
        return self.child

    def backward(self,child):

        assert (child is not None)
        self.__class__.grad_func(self.parents[0],self.parents[1],child)
