import itertools
from collections import OrderedDict,namedtuple

import zorch
from ..parameter import Parameter


class _IncompatibleKeys(namedtuple('IncompatibleKeys', ['missing_keys', 'unexpected_keys'])):
    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return '<All keys matched successfully>'
        return super(_IncompatibleKeys, self).__repr__()

    __str__ = __repr__


def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


class Module:

    def __init__(self):

        super().__setattr__('training',True)
        super().__setattr__('_parameters',OrderedDict())
        super().__setattr__('_buffers',OrderedDict())
        super().__setattr__('_modules',OrderedDict())

    def register_buffer(self,name,tensor):
        if '_buffers' not in self.__dict__:
            raise AttributeError("cannot assign buffer before Module.__init__() call")
        elif '.' in name:
            raise KeyError("buffer can't contain \".\"")
        elif name == '':
            raise KeyError("buffer name can't be empty string\"\"")
        elif hasattr(self,name) and name not in self._buffers:
            raise KeyError("attribute '{}' already exists".format(name))
        elif tensor is not None and not isinstance(tensor,zorch.Tensor):
            raise TypeError("cannot assign {} object to buffer '{}' zorch Tensor or None required".format(type(tensor),name))
        else:
            self._buffers[name] = tensor

    def register_parameter(self,name,param):
        if '_parameters' not in self.__dict__:
            raise AttributeError("cannot assign parameter before Module.__init__() call")
        elif '.' in name:
            raise KeyError("paramter name can't contain \".\"")
        elif name == '':
            raise KeyError("parameter name can't be empty string\"\"")
        elif hasattr(self,name) and name not in self._parameters:
            raise KeyError("attribute '{}' already exists".format(name))
        if param is None:
            self._parameters[name] = None
        elif not isinstance(param,Parameter):
            raise TypeError("cannot assign '{}' object to parameter '{}' "
                            "(torch.nn.Parameter or None required)"
                            .format(type(param), name))
        elif param.op:
            raise ValueError("Cannot assign non-leaf Tensor to parameter '{0}'. Model "
                            "parameters must be created explicitly. To express '{0}' "
                            "as a function of another Tensor, compute the value in "
                            "the forward() method.".format(name))
        else:
            self._parameters[name] = param

    def add_module(self,name,module):
        if not isinstance(module,Module) and module is not None:
            raise TypeError("{} is not a Module subclass".format(module))
        elif hasattr(self,name) and name not in self._modules:
            raise KeyError("attribute '{}' already exists".format(name))
        elif '.' in name:
            raise KeyError("module name can't contain \".\", got: {}".format(name))
        elif name == '':
            raise KeyError("module name can't be empty string \"\"")
        self._modules[name] = module

    def __getattr__(self,name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, name))

    def __setattr__(self,name,value):
        if isinstance(value,Parameter):
            self._parameters[name] = value
        elif isinstance(value,Module):
            self._modules[name] = value
        else:
            buffers = self.__dict__.get('_buffers')
            if buffers is not None and name in buffers:
                if value is not None and not isinstance(value,zorch.Tensor):
                    raise TypeError("cannot assign '{}' as buffer '{}' "
                                    "(Tensor or None expected)"
                                    .format(type(value), name))
                buffers[name] = value
            else:
                super().__setattr__(name,value)

    def __delattr__(self,name):
        if name in self._parameters:
            del self._parameters[name]
        elif name in self._buffers:
            del self._buffers[name]
        elif name in self._modules:
            del self._modules[name]
        else:
            super().__delattr__(name)

    def _check_shape_mismatch(self,input_param,param):
        if input_param.shape != param.shape:
            return True

    def _load_from_state_dict(self,state_dict,prefix,strict,missing_keys,unexpected_keys,error_msgs):
        persistent_buffers = {k:v for k,v in self._buffers.items()}
        local_name_params = itertools.chain(self._parameters.items(),persistent_buffers.items())
        local_state = {k:v for k,v in local_name_params if v is not None}

        for name,param in local_state.items():
            key = prefix + name
            if key in state_dict:
                input_param = state_dict[key]
                if self._check_shape_mismatch(input_param,param):
                    error_msgs.append('size mismatch for {}: copying a param with shape {} from checkpoint, '
                                      'the shape in current model is {}.'
                                      .format(key, input_param.shape, param.shape))
                    continue
                try:
                    param.do_copy(input_param)
                except Exception as ex:
                    error_msgs.append('While copying the parameter named "{}", '
                                      'whose dimensions in the model are {} and '
                                      'whose dimensions in the checkpoint are {}, '
                                      'an exception occurred : {}.'
                                      .format(key, param.data.size, input_param.data.size, ex.args))

            elif strict:
                missing_keys.append(key)

        if strict:
            for key in state_dict.keys():
                if key.startswith(prefix):
                    input_name = key[len(prefix):]
                    input_name = input_name.split('.',1)[0]
                    if input_name not in self._modules and input_name not in local_state:
                        unexpected_keys.append(key)

    def load_state_dict(self,state_dict,strict=True):
        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        state_dict = state_dict.copy()

        def load(module,prefix=''):
            module._load_from_state_dict(state_dict, prefix, True, missing_keys, unexpected_keys, error_msgs)
            for name,child in module._modules.items():
                if child is not None:
                    load(child,prefix + name+ '.')

        load(self)
        del load

        if strict:
            if len(unexpected_keys) > 0:
                error_msgs.insert(0, 'Unexpected key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in unexpected_keys)))
            if len(missing_keys) > 0:
                error_msgs.insert(0, 'Missing key(s) in state_dict: {}. '.format(
                        ', '.join('"{}"'.format(k) for k in missing_keys)))

        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                self.__class__.__name__, "\n\t".join(error_msgs)))
        return _IncompatibleKeys(missing_keys, unexpected_keys)

    def _save_to_state_dict(self,destination,prefix):
        for name,param in self._parameters.items():
            if param is not None:
                param_data = param.data
            
                if param_data.__class__ is zorch.cp.ndarray :
                    destination[prefix + name] = param_data.get()
                else:
                    destination[prefix + name] = param_data

        for name,buf in self._buffers.ietms():
            if buf is not None:
                buf_data = buf.data
                destination[prefix + name] = buf_data.get() if buf_data.__class__ is cp.ndarray else buf_data

    def state_dict(self,destination=None,prefix='.'):
        if destination is None:
            destination = OrderedDict()
        self._save_to_state_dict(destination, prefix)
        for name, module in self._modules.items():
            if module is not None:
                module.state_dict(destination, prefix + name + '.')
        return destination

    def _get_name(self):
        return self.__class__.__name__

    def extra_repr(self):
        return ''

    def __repr__(self):
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str

    def forward(*args,**kwargs):
        raise NotImplementedError

    def __call__(self,*args,**kwargs):
        return self.forward(*args,**kwargs)

    def named_modules(self,memo=None,prefix=''):
        if memo is None:
            memo = set()
        if self not in memo:
            memo.add(self)
            yield prefix, self
            for name, module in self._modules.items():
                if module is None:
                    continue
                submodule_prefix = prefix + ('.' if prefix else '') + name
                for m in module.named_modules(memo, submodule_prefix):
                    yield m

    def modules(self):
        for name,module in self.named_modules():
            yield module


    def named_children(self):
        memo = set()
        for name, module in self._modules.items():
            if module is not None and module not in memo:
                memo.add(module)
                yield name, module

    def children(self):
        for name,module in self.named_children():
            yield module


    def _named_members(self,get_members_fn,prefix='',recurse=True):
        memo = set()
        modules = self.named_modules(prefix=prefix) if recurse else [(prefix, self)]
        for module_prefix, module in modules:
            members = get_members_fn(module)
            for k, v in members:
                if v is None or v in memo:
                    continue
                memo.add(v)
                name = module_prefix + ('.' if module_prefix else '') + k
                yield name, v

    def parameters(self,recurse=True):
        for name,param in self.named_parameters(recurse=recurse):
            yield param

    def named_parameters(self,prefix='',recurse=True):
        gen = self._named_members(
            lambda module: module._parameters.items(),
            prefix=prefix, recurse=recurse)
        for elem in gen:
            yield elem

    def train(self,mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def apply(self,fn):
        for module in self.children():
            module.apply(fn)
        #print('first')
        fn(self)
        return self




























