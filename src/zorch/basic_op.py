from .op import UnaryOp,BinaryOp
from . import func as f


class add(BinaryOp):
    func = f.Add
    grad_func = f.AddBackward

class sub(BinaryOp):
    func = f.Sub
    grad_func = f.SubBackward

class neg(UnaryOp):
    func = f.Neg
    grad_func = f.NegBackward

class pow(BinaryOp):
    func = f.Pow
    grad_func = f.PowBackward

class div(BinaryOp):
    func = f.Div
    grad_func = f.DivBackward

class mult(BinaryOp):
    func = f.Mult
    grad_func = f.MultBackward

class exp(UnaryOp):
    func = f.Exp
    grad_func = f.ExpBackward

class mm(BinaryOp):
    func = f.Mm
    grad_func = f.MmBackward

class transpose(UnaryOp):
    func = f.Transpose
    grad_func = f.TransposeBackward

class sum(UnaryOp):
    func = f.Sum
    grad_func = f.SumBackward

class mean(UnaryOp):
    func = f.Mean
    grad_func = f.MeanBackward

class var(UnaryOp):
    func = f.Var
    grad_func = f.VarBackward

class concat(UnaryOp):
    func = f.Concat
    grad_func = f.ConcatBackward

class slice(UnaryOp):
    func = f.Slice
    grad_func = f.SliceBackward

class repeat(UnaryOp):
    func = f.Repeat
    grad_func = f.RepeatBackward

class view(UnaryOp):
    func = f.View
    grad_func = f.ViewBackward

class expand(UnaryOp):
    func = f.Expand
    grad_func = f.ExpandBackward

class squeeeze(UnaryOp):
    func = f.Squeeze
    grad_func = f.SqueezeBackward

class unsqueeze(UnaryOp):
    func = f.Unsqueeze
    grad_func = f.UnsqueezeBackward

class maskedfill(UnaryOp):
    func = f.MaskedFill
    grad_func = f.MaskedFillBackward

