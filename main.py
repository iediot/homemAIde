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

x = Value(50.0)
w = Value(2.0)
b = Value(10.0)

lr = 0.00001
target = Value(212.0)

for step in range(20):
    # 1. forward: predict and measure wrongness
    wx = w * x
    pred = wx + b
    error = pred - target
    loss = error * error

    # 2. reset grads (they use +=, so old values would pile up)
    w.grad = 0.0
    b.grad = 0.0

    # 3. backward: your four lines, with loss.grad = 1.0 first
    loss.grad = 1.0
    loss._backward()
    error._backward()
    pred._backward()
    wx._backward()

    # 4. learn: nudge each knob against its gradient
    w.data -= lr * w.grad
    b.data -= lr * b.grad

    print(step, loss.data, w.data, b.data)

print(loss, w.grad, b.grad)