"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(self.in_features, self.out_features))
        self.bias = Parameter(ops.transpose(init.kaiming_uniform(self.out_features, 1))) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        X = ops.matmul(X, self.weight)
        if self.bias:
            X = ops.add(X, ops.broadcast_to(self.bias, X.shape))
        return X
        ### END YOUR SOLUTION



class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        if len(X.shape) == 1:
            return X
        n = 1
        for i in X.shape[1:]:
            n *= i
        return ops.reshape(X, (X.shape[0], n))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        x = ops.relu(x)
        return x
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        num, classes = logits.shape
        y_one_hot = init.one_hot(classes, y)
        loss = (ops.log(ops.exp(logits)) - ops.broadcast_to(
                ops.reshape(ops.logsumexp(logits, 1), (num, 1)),
            logits.shape)) * y_one_hot
        loss = ops.divide_scalar(loss, -num)
        loss = ops.summation(loss)
        return loss
        ### END YOUR SOLUTION



class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(Tensor([1.0] * dim, device=device, dtype=dtype))
        self.bias = Parameter(Tensor([0.0] * dim, device=device, dtype=dtype))
        self.running_mean = Tensor([0.0] * dim, device=device, dtype=dtype)
        self.running_var = Tensor([1.0] * dim, device=device, dtype=dtype)
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch = x.shape[0]
        if self.training:
            mean = ops.summation(x, 0) / batch
            self.running_mean = self.momentum * mean + \
                (1 - self.momentum) * self.running_mean
            mean = ops.broadcast_to(ops.reshape(mean, (1, self.dim)), x.shape)

            std = ops.summation(ops.power_scalar(x - mean, 2), 0) / batch
            self.running_var = self.momentum * std + \
                (1 - self.momentum) * self.running_var
            std = ops.broadcast_to(ops.reshape(std, (1, self.dim)), x.shape)

            x = (x - mean) / ops.power_scalar(std + self.eps, 0.5) * \
                ops.broadcast_to(ops.reshape(self.weight, (1, self.dim)), x.shape) \
                + ops.broadcast_to(ops.reshape(self.bias, (1, self.dim)), x.shape)
            return x
        else:
            x = (x - self.running_mean) / ((self.running_var + self.eps) ** 0.5)
            return x
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(Tensor([1.0] * dim, device=device, dtype=dtype))
        self.bias = Parameter(Tensor([0.0] * dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        batch = x.shape[0]
        mean = ops.broadcast_to(
            ops.reshape(ops.summation(x, 1) / self.dim, (batch, 1)),
            x.shape)
        std = ops.broadcast_to(ops.reshape(
            ops.summation(
                ops.power_scalar(x - mean, 2), 1
            ) / self.dim, (batch, 1)), x.shape)
        x = (x - mean) / ops.power_scalar(std + self.eps, 0.5) * \
            ops.broadcast_to(ops.reshape(self.weight, (1, self.dim)), x.shape) + \
            ops.broadcast_to(ops.reshape(self.bias, (1, self.dim)), x.shape)
        return x
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training and self.p > 0.0:
            shape = x.shape
            mask = init.randb(*shape, p=(1-self.p), dtype='float32', device=x.device)
            x = ops.mul_scalar(ops.multiply(mask, x), 1 / (1 - self.p))
        return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.add(x, self.fn(x))
        ### END YOUR SOLUTION



