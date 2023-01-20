from zorch import xp,cp,np,is_grad_enable,no_grad
from . import basic_op as bp



class Tensor:

    """
    nd_array wrapper support autograd

    """

    def __init__(self,data,requires_grad=False,**kwargs):

        self.data = Tensor.__array_(data)
        self.requires_grad = requires_grad and is_grad_enable()
        self.op = None
        self.grad = xp.array([0.]*self.data.size,dtype=xp.float32).reshape(self.shape) if self.requires_grad else xp.uint8(0)
        self.kwargs = kwargs


    def zero_grad(self):
        self.grad = xp.array([0.]*self.data.size,dtype=xp.float32).reshape(self.shape)

    @staticmethod
    def __array_(data):

        if type(data) in {bool,int,float,list,np.ndarray,cp.ndarray}:
            return xp.array(data,dtype=np.float32)
        else:
            raise TypeError(f"expected data to be one of {bool,int,float,list,np.ndarray,cp.ndarray} got {data}")
    
    @staticmethod
    def _assert_tensor(t):
        
        if not isinstance(t,Tensor):
            t = Tensor(t)
        return t
        
    @property
    def requires_grad(self):
        return self._requires_grad

    @requires_grad.setter
    def requires_grad(self,value):
        
        if not isinstance(value,bool):
            raise RuntimeError("requires_grad must be a bool")
        
        if value is True:
            dtype = self.data.dtype.type
            if not issubclass(dtype,np.complexfloating) and not issubclass(dtype,np.floating):
                raise RuntimeError("zorch autograd engine only support Tensor of floating and complex data type ")
        self._requires_grad = value

    def do_copy(self,_tensor):

        if not isinstance(_tensor,Tensor):
            raise RuntimeError(" input  must be a Tensor got {_tensor}")
        if _tensor.shape != self.shape:
            raise RuntimeError(f" The size of tensor a {self.shape} must match the size of tensor b{_tensor.shape}")
        data = _tensor.data.copy()
        _from = data.__class__
        _to = self.data.__class__

        if _to is np.ndarray:
            if _from is not np.ndarray:
                data = data.get()
        else:
            if _from is np.ndarray:
                data = cp.ndarray(data)
        self.data = data
        return self

    def astype(self,dtype,**kwargs):
        self.data = self.data.astype(dtype,**kwargs)
        return self
    
    @property
    def device(self): return 'cpu' if self.data.__class__ == np.ndarray else 'cuda'

    @property
    def shape(self): return self.data.shape

    @shape.setter
    def shape(self,new_shape): self.data.shape = new_shape

    @property
    def ndim(self): return self.data.ndim
    
    @property
    def dtype(self): return self.data.dtype

    @property
    def strides(self): return self.data.strides

    @strides.setter
    def strides(self,new_strides): self.data.strides = new_strides

    def numel(self): return self.data.size

    def detach(self): return Tensor(self.data,requires_grad=False,**self.kwargs)

    def contiguous(self):
        self.data = xp.ascontiguousarray(self.data)
        return self
    
    def item(self):
        return self.data.item()


    def numpy(self):
        data = self.data

        if self.requires_grad:
            raise RuntimeError(f"Can't call numpy() on a Tensor that requires_grad ,Use tensor.detach().numpy()")

        if data.__class__ is cp.ndarray:
            data = data.get()

        return data

    def unaryop(self,operation,input0,*args,**kwargs):
        op = operation(input0)
        output = op.forward(*args,**kwargs)
        output.op = op
        return output

    def binaryop(self,operation,input0,input1,*args,**kwargs):
        op = operation(input0,input1)
        output = op.forward(*args,**kwargs)
        output.op = op
        return output

    @staticmethod
    def make_unaryop(operation,input0,*args,**kwargs):
        op = operation(input0)
        output = op.forward(*args,**kwargs)
        output.op = op
        return output

    @staticmethod
    def make_binaryop(operation,input0,input1,*args,**kwargs):
        op = operation(input0,input1)
        output = op.forward(*args,**kwargs)
        output.op = op
        return output

    # toposort and backpropagation

    def toposort(self):

        def _toposort(node,visited,nodes):
            visited.add(node)
            if node.op:
                [_toposort(i,visited,nodes) for i in node.op.parents if i not in visited]
                nodes.append(node)
            return nodes
        return _toposort(self,set(),[])

    def backward(self):

        assert self.shape == (),"Can't call backward on a no scalar function!!"

        self.grad = xp.ones(self.shape)

        for t in reversed(self.toposort()):
            if not any(x.requires_grad for x in t.op.parents):continue
            t.op.backward(t.op.child)
            t.op = None

    def __repr__(self):

        device_id = cp.cuda.runtime.getDevice() if xp is cp else ""
        grad_fn = f",grad_fn=<{self.op.grad_func.__name__}{device_id}>" if self.op else ''
        requires_grad = f", requires_grad={self.requires_grad}" if self.op is  None and self.requires_grad else ''
        if self.data is None:
            return str(None)
        else:
            s = np.array2string(self.data, separator=', ', precision=4).replace('\n', '\n' + ' ' * 7)
        return f"tensor({s}{grad_fn}{requires_grad})"

    def __len__(self):
        return self.shape[0]

    def __add__(self,other):
        other = Tensor._assert_tensor(other)
        return self.binaryop(bp.add,self,other)

    __radd__ = __add__
    
    def __iadd__(self,other):
        assert not self.requires_grad,"In-place operation is forbidden in node requires grad"

        if isinstance(other,Tensor):
            other = other.data
        self.data += other
        return self

    def __sub__(self,other):
        other = Tensor._assert_tensor(other)
        return self.binaryop(bp.sub,self,other)

    def __rsub__(self,other):
        other = Tensor._assert_tensor(other)
        return self.binaryop(bp.sub,other,self)

    def __isub__(self,other):
        assert not self.requires_grad,"In-place operation is forbidden in node requires grad"

        if isinstance(other,Tensor):
            other = other.data
        self.data -= other
        return self

    def __neg__(self):
        return self.unaryop(bp.neg,self)

    def __mul__(self,other):
        other = Tensor._assert_tensor(other)
        return self.binaryop(bp.mult,self,other)

    __rmul__ = __mul__

    def __imul__(self,other):
        assert not self.requires_grad,"In-place operation is forbidden in node requires grad"

        if isinstance(other,Tensor):
            other = other.data
        self.data *= other
        return self

    def __truediv__(self,other):
        other = Tensor._assert_tensor(other)
        return self.binaryop(bp.div,self,other)

    def __rtruediv__(self,other):
        other = Tensor._assert_tensor(other)
        return self.binaryop(bp.div,other,self)

    def __itruediv__(self,other):
        assert not self.requires_grad,"In-place operation is forbidden in node requires grad"

        if isinstance(other,Tensor):
            other = other.data
        self.data /= other
        return self

    def __pow__(self,other):
        other = Tensor._assert_tensor(other)
        return self.binaryop(bp.pow,self,other)

    def __rpow__(self,other):
        other = Tensor._assert_tensor(other)
        return self.binaryop(bp.pow,other,self)

    def __matmul__(self,other):
        other = Tensor._assert_tensor(other)
        return self.binaryop(bp.mm,self,other)

    def __imatmul__(self,other):
        assert not self.requires_grad,"In-palce operation is forbidden in node requires grad"

        if isinstance(other,Tensor):
            other = other.data
        self.data *= other
        return self

    @no_grad()
    def __lt__(self,other):
        if isinstance(other,Tensor):
            other = other.data
        return Tensor(self.data < other)

    @no_grad()
    def __le__(self,other):
        if isinstance(other,Tensor):
            other = other.data
        return Tensor(self.data <= other)

    @no_grad()
    def __gt__(self,other):
        if isinstance(other,Tensor):
            other = other.data
        return Tensor(self.data > other)

    @no_grad()
    def __ge__(self,other):
        if isinstance(other,Tensor):
            other = other.data
        return Tensor(self.data >= other)

    @no_grad()
    def eq(self,other):
        if isinstance(other,Tensor):
            other = other.data
        return Tensor(self.data == other)

    @no_grad()
    def ne(self,other):
        if isinstance(other,Tensor):
            other = other.data
        return Tensor(self.data != other)

    def exp(self):
        return self.unaryop(bp.exp,self)

    def transpose(self,axes=None):
        return self.unaryop(bp.transpose,self,axes=axes)

    def T(self,axes=None):
        return self.transpose(self,axes=axes)

    def mean(self,axis=None,keepdims=False):
        return self.unaryop(bp.mean,self,axis=axis,keepdims=keepdims)

    def sum(self,axis=None,keepdims=False):
        return self.unaryop(bp.sum,self,axis=axis,keepdims=keepdims)

    def var(self,axis=None,ddof=1,keepdims=False):
        return self.unaryop(bp.var,self,axis=axis,ddof=ddof,keepdims=keepdims)

    def concat(self,other,axis=0):

        if not isinstance(other,list):
            raise TypeError(f"expected list of Tensor got {other}")
        parents = [self]
        parenst.extend(other)
        return self.unaryop(bp.concat,parents,axis=axis)
    
    def repeat(self,reps):
        return self.unaryop(bp.repeat,self,reps)

    def expand(self,*dims):
        return self.unaryop(bp.expand,self,*dims)

    def squeeze(self,axis=None):
        return self.unaryop(bp.squueze,self,axis=axis)

    def unsqueeze(self,axis):
        return self.unaryop(bp.unsqueeze,self,axis)

    def maskedfill(self,mask,value):
        return self.unaryop(bp.maskedfill,self,mask,value)

    def view(self,shape):
        return self.unaryop(bp.view,self,shape)

    def slice(self,key):
        return self.unaryop(bp.slice,self,key=key)

    def __getitem__(self,key):
        return self.slice(key)

    def __setitem__(self,key,value):
        self.data[key]= value

    def argmax(self,axis=None,keepdims=False):
        return Tensor(xp.argmax(self.data,axis=axis,keepdims=keepdims)).astype(np.uint8)

    def argmin(self,axis=None,keepdims=False):
        return Tensor(xp.argmin(self.data,axis=axis,keepdims=keepdims)).astype(np.uint8)

    

