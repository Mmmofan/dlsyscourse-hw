"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad * \
            self.scalar * \
            power_scalar(node.inputs[0], self.scalar - 1),)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return (out_grad / rhs,
                negate(out_grad) * lhs / (power_scalar(rhs, 2)))
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad / self.scalar,)
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        dim = list(range(a.ndim))
        if self.axes is None:
            dim[-1], dim[-2] = dim[-2], dim[-1]
        else:
            dim[self.axes[0]], dim[self.axes[1]] = \
                dim[self.axes[1]], dim[self.axes[0]]
        return a.permute(dim)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (transpose(out_grad, self.axes),)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        shape = node.inputs[0].shape
        return (out_grad.reshape(shape),)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return a.broadcast_to(self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        base_shape = [1] * (len(self.shape) - len(input_shape)) + list(input_shape)
        axes = []
        for i in range(len(base_shape)):
            if self.shape[i] != base_shape[i]:
                axes.append(i)
        out_grad = summation(out_grad, axes=tuple(axes))
        return (out_grad.reshape(input_shape),)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.sum(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        base_shape = list(input_shape)
        if isinstance(self.axes, int): self.axes = (self.axes,)
        axes = list(range(len(base_shape))) if self.axes is None else self.axes
        for ax in axes:
            base_shape[ax] = 1
        out_grad = out_grad.reshape(base_shape)
        out_grad = out_grad.broadcast_to(input_shape)
        return (out_grad,)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        a_shape = a.shape
        b_shape = b.shape
        lhs = out_grad @ transpose(b)
        rhs = transpose(a) @ out_grad
        while a_shape != lhs.shape:
            lhs = lhs.sum(0)
        while b_shape != rhs.shape:
            rhs = rhs.sum(0)
        return (lhs, rhs)
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (-out_grad,)
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.log()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad / node.inputs[0],)
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.exp()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (out_grad * node.inputs[0].exp(),)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        b = a.maximum(0)
        return b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        inp = node.inputs[0]
        mask = inp.maximum(0) / inp.maximum(0)
        return (out_grad * mask,)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class Sigmoid(TensorOp):
    def compute(self, a):
        return 1 / (1  + -a.exp())

    def gradient(self, out_grad, node):
        inp = node.inputs[0]
        return (out_grad * sigmoid(inp) * (1 - sigmoid(inp)),)


def sigmoid(a):
    return Sigmoid()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        self.mZ = Z.max(self.axes, keepdims=True)
        mZ = self.mZ.broadcast_to(Z.shape)
        Z_ = (Z - mZ).exp()
        Z_ = Z_.sum(self.axes)
        Z_ = Z_.log()
        return Z_ + self.mZ.reshape(Z_.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        inp = node.inputs[0]
        input_shape = inp.shape
        mZ = Tensor(self.mZ.broadcast_to(input_shape), device=inp.device)
        base_shape = list(input_shape)
        if isinstance(self.axes, int): self.axes = (self.axes,)
        axes = list(range(len(base_shape))) \
            if self.axes is None else self.axes
        for ax in axes:
            base_shape[ax] = 1
        out_grad = out_grad / summation(exp((inp - mZ)), self.axes)
        out_grad = out_grad.reshape(base_shape)
        out_grad = out_grad.broadcast_to(input_shape)
        out_grad = out_grad * exp(inp - mZ)
        return (out_grad,)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.tanh()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        inp = node.inputs[0]
        out_grad = out_grad * (
            1 + (-tanh(inp) ** 2)
        )
        return (out_grad,)
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args: TensorTuple) -> Tensor:
        ### BEGIN YOUR SOLUTION
        shape = args[0].shape
        dim = len(shape)
        num = len(args)
        assert self.axis <= dim, f"axis {self.axis} is out of bound for array of dimension {dim}"
        tmp_shape = [num] + list(shape)
        output = args[0].broadcast_to(tmp_shape).compact()
        out_dim = list(range(1, dim+1))
        out_dim.insert(self.axis, 0)
        for i in range(num):
            assert args[i].shape == shape, "all input arrays must have the same shape"
            output[i] = args[i]
        output = output.permute(out_dim).compact()
        return output
        ### END YOUR SOLUTION


    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out_grad = split(out_grad, self.axis)
        return out_grad
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        dim = A.ndim
        tmp_dim = list(range(0, dim))
        tmp_dim.remove(self.axis)
        tmp_dim = [self.axis] + tmp_dim
        B = A.permute(tmp_dim)
        output = []
        for i in range(B.shape[0]):
            out = B[i].compact()
            out = out.reshape(list(out.shape)[1:]).compact()
            output.append(out)
        return tuple(output)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out_grad = stack(out_grad, self.axis)
        return (out_grad,)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (flip(out_grad, self.axes))
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)



class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        shape = a.shape
        out_shape = list(shape)
        slices = [slice(0, out_shape[idx]) for idx in range(len(shape))]
        for ax in self.axes:
            if ax >= len(out_shape):
                continue
            out_shape[ax] = out_shape[ax] * (1 + self.dilation)
            slices[ax] = slice(0, out_shape[ax], 1 + self.dilation)
        out_tensor = NDArray.make(out_shape, device=a.device)
        out_tensor.fill(0)
        out_tensor[tuple(slices)] = a
        return out_tensor
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out_grad = undilate(out_grad, self.axes, self.dilation)
        return (out_grad,)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)

class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        shape = a.shape
        slices = [slice(0, shape[idx]) for idx in range(len(shape))]
        for ax in self.axes:
            if ax >= len(shape):
                continue
            slices[ax] = slice(0, shape[ax], 1 + self.dilation)
        return a[tuple(slices)].compact()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out_grad = dilate(out_grad, self.axes, self.dilation)
        return (out_grad,)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        N, H, W, C_in = A.shape
        K, _, I, C_out = B.shape
        assert C_in == I, "input tensor shape and kernel dosen't match"

        _A = A.pad((
            (0, 0),
            (self.padding, self.padding),
            (self.padding, self.padding),
            (0, 0))
        ) if self.padding > 0 else A

        inner_dim = K * K * C_in
        Ns, Hs, Ws, Cs = _A.strides
        H_out = (H - K + 2 * self.padding) // self.stride + 1
        W_out = (W - K + 2 * self.padding) // self.stride + 1

        _A = _A.as_strided(
            shape=(N, H_out, W_out, K, K, C_in),
            strides=(Ns, Hs*self.stride, Ws*self.stride, Hs, Ws, Cs)
        ).compact().reshape((-1, inner_dim))
        out = _A @ B.reshape((-1, C_out))

        return out.reshape((N, H_out, W_out, C_out))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)



