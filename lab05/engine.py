"""Lab 5, Tasks 1-3: scalar autograd engine.
Value stores data, grad, _parents, _op and _backward (local derivative rule).
Operators build the graph in the forward pass; backward() runs it in reverse topological order."""
import math
import random


class Value:
    def __init__(self, data, _parents=(), _op="", label=""):
        self.data = float(data)
        self.grad = 0.0
        self._parents = tuple(_parents)
        self._op = _op
        self.label = label
        self._backward = lambda: None  # leaf nodes have no local rule

    def __repr__(self):
        return f"Value(data={self.data:.6g}, grad={self.grad:.6g}, op='{self._op}')"

    # Task 2: core operations (each one adds a node to the graph) 

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            # d(a+b)/da = 1, d(a+b)/db = 1
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            # d(a*b)/da = b, d(a*b)/db = a
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, k):
        assert isinstance(k, (int, float)), "only constant int/float exponents are supported"
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            # d(x^k)/dx = k * x^(k-1)
            self.grad += k * (self.data ** (k - 1)) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward():
            # derivative is 1 for x > 0, else 0 (PyTorch also uses 0 at x == 0)
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad

        out._backward = _backward
        return out

    #  Task 5 extra operations needed for sigmoid + BCE 

    def sigmoid(self):
        x = self.data
       
        s = 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))
        out = Value(s, (self,), "sigmoid")

        def _backward():
            self.grad += s * (1.0 - s) * out.grad

        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad

        out._backward = _backward
        return out

    
    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return (-self) + other

    def __radd__(self, other):  # other + self
        return self + other

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):
        return self * (other ** -1 if isinstance(other, Value) else 1.0 / other)

    #  Task 3: backward pass 

    def topological_order(self):
        """Return all nodes so that every node comes AFTER all of its parents.
        Iterative DFS (no recursion limit problems on big graphs)."""
        order, visited = [], set()
        stack = [(self, False)]
        while stack:
            node, children_done = stack.pop()
            if children_done:
                order.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))  
            for p in node._parents:
                if p not in visited:
                    stack.append((p, False))
        return order

    def backward(self):
        order = self.topological_order()
        # 1) d(out)/d(out) = 1
        self.grad = 1.0
        # 2) visit nodes in REVERSE topological order, so a node's grad is
        #    complete before it is passed on to its parents.
        # 3) every _backward uses "+=", so a node used several times
        #    accumulates the gradient from all of its uses.
        for node in reversed(order):
            node._backward()


def vsum(values):
    """Sum a list of Values (like tensor.sum())."""
    total = Value(0.0)
    for v in values:
        total = total + v
    return total


def zero_grad(params):
    for p in params:
        p.grad = 0.0


# ---------- Tiny MLP built from Values (used in Task 5) ----------

class MLP:
    """2 -> hidden (ReLU) -> 1 (sigmoid). Weights can be loaded from lists so
    the same initial weights can be copied into PyTorch."""

    def __init__(self, W1, b1, W2, b2):
        self.W1 = [[Value(w) for w in row] for row in W1]   # [hidden][in]
        self.b1 = [Value(b) for b in b1]                    # [hidden]
        self.W2 = [[Value(w) for w in row] for row in W2]   # [1][hidden]
        self.b2 = [Value(b) for b in b2]                    # [1]

    def parameters(self):
        ps = [w for row in self.W1 for w in row] + self.b1
        ps += [w for row in self.W2 for w in row] + self.b2
        return ps

    def __call__(self, x):
        h = [vsum(w * xi for w, xi in zip(row, x)) + b for row, b in zip(self.W1, self.b1)]
        h = [v.relu() for v in h]
        z = vsum(w * hi for w, hi in zip(self.W2[0], h)) + self.b2[0]
        return z.sigmoid()


def bce(p, y):
    """Binary cross-entropy for one sample: -(y log p + (1-y) log(1-p))."""
    if y == 1:
        return -(p.log())
    return -((1 - p).log())


if __name__ == "__main__":
    # small demo: print the graph of a tiny expression
    random.seed(0)
    a, b, c = Value(2.0, label="a"), Value(-3.0, label="b"), Value(10.0, label="c")
    out = ((a * b + c) ** 2).relu()
    out.backward()
    for n in out.topological_order():
        print(f"{n.label or n._op:>6}  data={n.data:8.3f}  grad={n.grad:8.3f}")