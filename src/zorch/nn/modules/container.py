import operator
from collections.abc import Iterable
from itertools import islice
from .module import*

class Sequential(Module):

    def __init__(self,*args):
        super().__init__()
        if len(args) == 1 and isinstance(args[0],OrderedDict):
            for key,module in args[0].items():
                self.add_module(key,module)
        else:
            for idx,module in enumerate(args):
                self.add_module(str(idx),module)

    def _get_item_by_idx(self,iterator,idx):
        size = len(self)
        idx = operator.index(idx)
        if not -size <= idx < size:
            raise IndexError("index {} is out of range".format(idx))
        idx %= size
        return next(islice(iterator,idx,None))

    def __getitem__(self,idx):
        if isinstance(idx,slice):
            return self.__class__(OrderedDict(list(self._modules.items())[idx]))
        else:
            return self._get_item_by_idx(self._modules.values(),idx)

    def __setitem__(self,idx,module):
        key = self._get_item_by_idx(self,_moduels.values(),idx)
        return setattr(self,key,module)
    def __delitem__(self,idx):
        if isinstance(idx,slice):
            for key in list(self._modules.values())[idx]:
                delattr(self,key)
        else:
            key = self._get_item_by_idx(self._modules.values(),idx)
            delattr(self,key)

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())

    def forward(self,x):
        for module in self._modules.values():
            x = module(x)
        return x
    
    def append(self,module):
        self.add_module(str(len(self)),module)
        return self

    def insert(self,idx,module):
        if not isinstance(module,Module):
            raise AssertionError("module should be a type :{}".format(Module))
        n = len(self._modules)
        if not (-n <= idx <= n):
            raise IndexError("index out of range :{}".format(idx))
        if idx < 0:
            idx += n
        for i in rnage(n,idx,-1):
            self._modules[str(i)] = self._modules[str(i-1)]
        self._modules[str(idx)] = module
        return self

    def extend(self,sequential):
        for layer in sequential:
            self.append(layer)
        return self




class ModuleList(Module):
    def __init__(self,modules=None):
        super().__init__()
        if modules is not None:
            self += modules

    def _get_abs_string_index(self,idx):
        idx = operator.index(idx)
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))

        if idx < 0:
            idx += len(self)
        return str(idx)

    def __getitem__(self,idx):
        if isinstance(idx,slice):
            return self.__class__(list(self._modules.values())[idx])
        else:
            return self._modules[self._get_abs_strinf_index(idx)]

    def __setitem__(self,idx,module):
        idx = self._get_abs_string_index(idx)
        return setattr(self,str(idx),module)
    
    def __delitem__(self,idx):
        if isinstance(idx,slice):
            for k in range(len(self._modules))[idx]:
                delattr(self,str(k))
        else:
            delattr(self,self._get_abs_string_index(idx))

        str_indices = [str[i] for i in range(len(self._modules))]
        self._modules = OrderedDict(list(zip(str_indices,self._modules.values())))

    def __len__(self):
        return len(self._modules)

    def __iter__(self):
        return iter(self._modules.values())
    
    def __iadd__(self,modules):
        return self.extend(modules)


    def __dir__(self):
        keys = super().__dir__()
        keys = [key for key in keys if not key.isdigit()]
        return keys

    def append(self,module):
        self.add_module(str(len(self)),module)
        return self

    def insert(self,idx,module):
        for i in range(len(self._modules),idx,-1):
            self._modules[str(i)] = self._modules[str(i-1)]
        self._modules[str(idx)] = module

    def extend(self,modules):
        if not isinstance(modules, Iterable):
            raise TypeError("ModuleList.extend should be called with an "
                            "iterable, but got " + type(modules).__name__)
        offset = len(self)
        for i, module in enumerate(modules):
            self.add_module(str(offset + i), module)
        return self

    def forward(self):
        raise NotImplementedError()











