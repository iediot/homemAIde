import math
import random
import numpy as np

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.grad = 0.0
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        out = Value(self.data - other.data, (self, other), '-')

        def _backward():
            self.grad += out.grad
            other.grad -= out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        self.grad = 1.0

        for v in reversed(topo):
            v._backward()

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out


class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1))
                  for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        act = self.b

        for wi, xi in zip(self.w, x):
            act += wi * xi

        return act.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        params = []

        for n in self.neurons:
            params.extend(n.parameters())

        return params


class MLP:
    def __init__(self, nin, nouts):
        sizes = [nin] + nouts
        self.layers = [Layer(sizes[i], sizes[i + 1])
                       for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):
        params = []

        for n in self.layers:
            params.extend(n.parameters())

        return params


class Tensor:
    def __init__(self, data, _children=(), _op=''):
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, data={self.data})"

    def backward(self):
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)

                for child in v._prev:
                    build(child)

                topo.append(v)

        build(self)

        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def __matmul__(self, other):
        out = Tensor(self.data @ other.data, (self, other), '@')

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def __add__(self, other):
        out = Tensor(self.data + other.data, (self, other), '+')

        def _backward():
            if self.data.shape == out.data.shape:
                self.grad += out.grad
            else:
                self.grad += out.grad.sum(axis=0, keepdims=True)

            if other.data.shape == out.data.shape:
                other.grad += out.grad
            else:
                other.grad += out.grad.sum(axis=0, keepdims=True)

        out._backward = _backward
        return out

    def __sub__(self, other):
        out = Tensor(self.data - other.data, (self, other), '-')

        def _backward():
            self.grad += out.grad
            other.grad -= out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        out = Tensor(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward
        return out

a = Tensor([[0.5, 3.0]])
b = Tensor([[2.0, 1.0]])
out = (a * b - b).tanh()
out.backward()
print(out)
print(a.grad)
print(b.grad)

# labels = {
#     "Iris-setosa": [1, -1, -1],
#     "Iris-versicolor": [-1, 1, -1],
#     "Iris-virginica": [-1, -1, 1]
# }
#
# all_xs = []
# all_ys = []
#
# with open("iris.data") as f:
#     for line in f:
#         line = line.strip()
#         if not line:
#             continue
#         parts = line.split(",")
#         measurement = [float(p) for p in parts[:4]]
#         species = labels[parts[-1]]
#         all_xs.append(measurement)
#         all_ys.append(species)
#
# xs = all_xs[0:40] + all_xs[50:90] + all_xs[100:140]
# ys = all_ys[0:40] + all_ys[50:90] + all_ys[100:140]
# xs_test = all_xs[40:50] + all_xs[90:100] + all_xs[140:150]
# ys_test = all_ys[40:50] + all_ys[90:100] + all_ys[140:150]
#
# maxes = []
# mins = []
#
# for col_idx in range(0, 4):
#     col = [row[col_idx] for row in xs]
#     maxes.append(max(col))
#     mins.append(min(col))
#
# def normalize(data):
#     result = []
#     for row in data:
#         new_row = []
#         for value, lo, hi in zip(row, mins, maxes):
#             new_row.append((value - lo) / (hi - lo))
#         result.append(new_row)
#     return result
#
# xs = normalize(xs)
# xs_test = normalize(xs_test)
# print(xs[0], xs_test[0])
#
# model = MLP(4, [8, 3])
#
# lr = 0.001
#
# for step in range(200):
#     loss = Value(0.0)
#     for x, y in zip(xs, ys):
#         out = model([Value(v) for v in x])
#         for o, t in zip(out, y):
#             error = o - Value(t)
#             loss += error * error
#
#     for p in model.parameters():
#         p.grad = 0.0
#
#     loss.backward()
#
#     for p in model.parameters():
#         p.data -= lr * p.grad
#
#     if step % 10 == 0:
#         print(step, loss.data)
#
# for x, y in zip(xs_test, ys_test):
#     out = model([Value(v) for v in x])
#     print(y, [round(o.data, 2) for o in out])