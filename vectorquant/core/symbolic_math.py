"""
Symbolic Math Engine (Basic AST Evaluator & Differentiator)
"""

class Expr:
    def eval(self, env):
        raise NotImplementedError
    def deriv(self, var):
        raise NotImplementedError
    def simplify(self):
        return self

class Var(Expr):
    def __init__(self, name):
        self.name = name
    def eval(self, env):
        return env[self.name]
    def deriv(self, var):
        if self.name == var:
            return Const(1.0)
        return Const(0.0)
    def __str__(self):
        return self.name

class Const(Expr):
    def __init__(self, val):
        self.val = val
    def eval(self, env):
        return self.val
    def deriv(self, var):
        return Const(0.0)
    def __str__(self):
        return str(self.val)

class Add(Expr):
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def eval(self, env):
        return self.left.eval(env) + self.right.eval(env)
    def deriv(self, var):
        return Add(self.left.deriv(var), self.right.deriv(var)).simplify()
    def simplify(self):
        l = self.left.simplify()
        r = self.right.simplify()
        if isinstance(l, Const) and l.val == 0.0:
            return r
        if isinstance(r, Const) and r.val == 0.0:
            return l
        if isinstance(l, Const) and isinstance(r, Const):
            return Const(l.val + r.val)
        return Add(l, r)
    def __str__(self):
        return f"({self.left} + {self.right})"

class Mul(Expr):
    def __init__(self, left, right):
        self.left = left
        self.right = right
    def eval(self, env):
        return self.left.eval(env) * self.right.eval(env)
    def deriv(self, var):
        # Product rule: u'v + uv'
        term1 = Mul(self.left.deriv(var), self.right)
        term2 = Mul(self.left, self.right.deriv(var))
        return Add(term1, term2).simplify()
    def simplify(self):
        l = self.left.simplify()
        r = self.right.simplify()
        if isinstance(l, Const) and l.val == 0.0:
            return Const(0.0)
        if isinstance(r, Const) and r.val == 0.0:
            return Const(0.0)
        if isinstance(l, Const) and l.val == 1.0:
            return r
        if isinstance(r, Const) and r.val == 1.0:
            return l
        if isinstance(l, Const) and isinstance(r, Const):
            return Const(l.val * r.val)
        return Mul(l, r)
    def __str__(self):
        return f"({self.left} * {self.right})"

class Power(Expr):
    def __init__(self, base, exponent):
        self.base = base
        self.exponent = exponent
    def eval(self, env):
        return self.base.eval(env) ** self.exponent.eval(env)
    def deriv(self, var):
        # General power rule (assuming exponent is Const for simplicity):
        # d/dx(x^n) = n * x^(n-1) * x'
        if isinstance(self.exponent, Const):
            n = self.exponent.val
            if n == 0:
                return Const(0.0)
            new_power = Power(self.base, Const(n - 1)).simplify()
            coef = Mul(self.exponent, new_power)
            return Mul(coef, self.base.deriv(var)).simplify()
        # Fallback if variable exponent (e^x etc not implemented fully here)
        return Const(0.0)
    def simplify(self):
        b = self.base.simplify()
        e = self.exponent.simplify()
        if isinstance(e, Const) and e.val == 0:
            return Const(1.0)
        if isinstance(e, Const) and e.val == 1:
            return b
        if isinstance(b, Const) and isinstance(e, Const):
            return Const(b.val ** e.val)
        return Power(b, e)
    def __str__(self):
        return f"({self.base}^{self.exponent})"

# =========================================================
# Automatic Mathematical Verification Engine
# =========================================================

def verify_identity(expr1, expr2, test_points=None, vars_tested=None):
    """
    Numerically checks if expr1 is functionally identical to expr2 over test_points.
    This acts as a theorem prover / sanity checker for derived equations vs expected.
    """
    if test_points is None:
        test_points = [{ 'x': 1.0, 'y': 2.0, 'S': 100.0, 'K': 100.0, 'r': 0.05, 'sigma': 0.2, 't': 1.0, 'T': 1.0 }]
        
    for env in test_points:
        try:
            val1 = expr1.eval(env)
            val2 = expr2.eval(env)
            if abs(val1 - val2) > 1e-6:
                return False
        except Exception:
            return False
            
    return True

# =========================================================
# Automatic Differentiation (Reverse-Mode AutoDiff)
# =========================================================

class ADNode:
    """
    Computational graph node for reverse-mode autodiff.
    """
    def __init__(self, value):
        self.value = value
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set()
        
    def __add__(self, other):
        other = other if isinstance(other, ADNode) else ADNode(other)
        out = ADNode(self.value + other.value)
        out._prev = {self, other}
        
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
        
    def __mul__(self, other):
        other = other if isinstance(other, ADNode) else ADNode(other)
        out = ADNode(self.value * other.value)
        out._prev = {self, other}
        
        def _backward():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        other = other if isinstance(other, ADNode) else ADNode(other)
        out = ADNode(self.value - other.value)
        out._prev = {self, other}
        
        def _backward():
            self.grad += out.grad
            other.grad -= out.grad
        out._backward = _backward
        return out
        
    def __truediv__(self, other):
        other = other if isinstance(other, ADNode) else ADNode(other)
        out = ADNode(self.value / other.value)
        out._prev = {self, other}
        
        def _backward():
            self.grad += (1.0 / other.value) * out.grad
            other.grad += (-self.value / (other.value ** 2)) * out.grad
        out._backward = _backward
        return out
        
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
                
        build_topo(self)
        
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

def ad_sin(node):
    import math
    out = ADNode(math.sin(node.value))
    out._prev = {node}
    def _backward():
        node.grad += math.cos(node.value) * out.grad
    out._backward = _backward
    return out

def ad_exp(node):
    import math
    out = ADNode(math.exp(node.value))
    out._prev = {node}
    def _backward():
        node.grad += out.value * out.grad
    out._backward = _backward
    return out

