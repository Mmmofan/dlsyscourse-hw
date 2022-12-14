"""The module.
"""
from typing import List
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
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(self.in_features, self.out_features, device=device, dtype=dtype))
        self.bias = Parameter(ops.transpose(init.kaiming_uniform(self.out_features, 1, device=device, dtype=dtype))) if bias else None
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


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.tanh(x)
        ### END YOUR SOLUTION


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.sigmoid(x)
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
        y_one_hot = init.one_hot(classes, y, device=y.device)
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


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


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
    def __init__(self, p=0.5):
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


class ConvBN(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.conv = Conv(in_channels, out_channels, kernel_size, stride, bias, device, dtype)
        self.bn = BatchNorm2d(out_channels, device=device, dtype=dtype)
        self.relu = ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.bn(self.conv(x)))


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.device = device
        self.dtype = dtype

        ### BEGIN YOUR SOLUTION
        self.padding = self.kernel_size // 2
        self.weight = Parameter(
            init.kaiming_uniform(
                kernel_size * kernel_size * in_channels,
                kernel_size * kernel_size * out_channels,
                shape=[kernel_size, kernel_size, in_channels, out_channels],
                dtype=dtype,
                device=device
            )
        )
        self.bias = None
        if bias:
            prob = 1.0 / (in_channels * kernel_size ** 2) ** 0.5
            self.bias = Parameter(
                init.rand(
                    out_channels,
                    low=-prob,
                    high=prob,
                    device=device,
                    dtype=dtype
                )
            )
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        x = ops.transpose(
            ops.transpose(x),
            (1, 3)
        ) # NCHW -> NCWH -> NHWC
        x = ops.conv(x, self.weight, stride=self.stride, padding=self.padding)
        if self.bias is not None:
            x = x + ops.broadcast_to(self.bias, x.shape)
        x = ops.transpose(
            ops.transpose(x, (1,3))
        ) # NHWC -> NCWH -> NCHW
        return x
        ### END YOUR SOLUTION


class RNNCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size: The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias: If False, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden weights of shape (input_size, hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        k = (1 / hidden_size) ** 0.5
        assert nonlinearity in ['relu', 'tanh']
        self.activate = ReLU() if nonlinearity == 'relu' else Tanh()
        self.W_ih = Parameter(
            init.rand(
                input_size,
                hidden_size,
                low=-k,
                high=k,
                device=device,
                dtype=dtype
            )
        )
        self.W_hh = Parameter(
            init.rand(
                hidden_size,
                hidden_size,
                low=-k,
                high=k,
                device=device,
                dtype=dtype,
            )
        )
        if bias:
            self.bias_ih = Parameter(
                init.rand(
                    hidden_size,
                    low=-k,
                    high=k,
                    device=device,
                    dtype=dtype,
                )
            )
            self.bias_hh = Parameter(
                init.rand(
                    hidden_size,
                    low=-k,
                    high=k,
                    device=device,
                    dtype=dtype
                )
            )
        else:
            self.bias_ih, self.bias_hh = None, None
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        X = X @ self.W_ih
        if self.bias_ih is not None:
            X = X + ops.broadcast_to(self.bias_ih, X.shape)
        if h is not None:
            X = X + h @ self.W_hh
        if self.bias_hh is not None:
            X = X + ops.broadcast_to(self.bias_hh, X.shape)
        X = self.activate(X)

        return X
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, nonlinearity='tanh', device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias - If False, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.rnn_cells = []
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.dtype = dtype
        self.nonlinearity = nonlinearity
        for layer_num in range(num_layers):
            if layer_num == 0:
                cell = RNNCell(
                    input_size,
                    hidden_size,
                    bias=bias,
                    nonlinearity=nonlinearity,
                    device=device,
                    dtype=dtype
                )
            else:
                cell = RNNCell(
                    input_size,
                    hidden_size,
                    bias=bias,
                    nonlinearity=nonlinearity,
                    device=device,
                    dtype=dtype
                )
            self.rnn_cells.append(cell)
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial
            hidden state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        seq_len, bs, input_size = X.shape
        if h0 is None:
            h0 = init.zeros(
                self.num_layers,
                bs,
                self.hidden_size,
                device=self.device,
                dtype=self.dtype
            )

        outputs = []
        X_split = ops.split(X, 0) # seq_len of [bs, input_size]
        h_split = list(ops.split(h0, 0)) # num_layer of [bs, hidden_size]
        for xi in X_split:
            for j, cell in enumerate(self.rnn_cells):
                hi = h_split[j]
                hi = cell(xi, hi)
                h_split[j] = hi
                xi = hi
            outputs.append(xi)
        outputs = ops.stack(outputs, 0)
        h0 = ops.stack(h_split, 0)

        return outputs, h0
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION
