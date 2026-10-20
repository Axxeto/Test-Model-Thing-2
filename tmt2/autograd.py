"""Autograd mini (numpy, float64) untuk TMT2.

Kontrak yang dipakai tests/test_autograd.py dan modul lain:
  - Tensor(data, requires_grad=False): .data, .grad, .backward(seed=None)
  - Param(data, name="", fast=False): Tensor daun; gradien dicatat per-task di p.tg[task]
    (task aktif = CTX.task). Graf boleh di-backward berkali-kali (graph reuse).
  - Daun Tensor biasa (requires_grad=True) mengakumulasi ke .grad (reset sendiri: t.grad = None).
"""
import numpy as np


class _Ctx:
    task = "default"


CTX = _Ctx()


def _unbroadcast(g, shape):
    shape = tuple(shape)
    if g.shape == shape:
        return g
    while g.ndim > len(shape):
        g = g.sum(axis=0)
    for ax, s in enumerate(shape):
        if s == 1 and g.shape[ax] != 1:
            g = g.sum(axis=ax, keepdims=True)
    return g


class Tensor:
    __array_ufunc__ = None  # numpy menyerahkan operator (ndarray * Tensor) ke Tensor

    def __init__(self, data, requires_grad=False, _parents=(), _fn=None):
        self.data = np.asarray(data, dtype=np.float64)
        self.requires_grad = bool(requires_grad)
        self.grad = None
        self._parents = _parents
        self._fn = _fn

    # -- info
    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor({self.data!r}, requires_grad={self.requires_grad})"

    # -- backprop
    def _accumulate(self, g):
        self.grad = g.copy() if self.grad is None else self.grad + g

    def backward(self, seed=None):
        if not self.requires_grad:
            return
        seed = np.ones_like(self.data) if seed is None else np.asarray(seed, dtype=np.float64)
        order, seen, stack = [], set(), [(self, False)]
        while stack:  # DFS iteratif: graf hasil unroll panjang tidak kena batas rekursi
            node, done = stack.pop()
            if done:
                order.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                if p.requires_grad and id(p) not in seen:
                    stack.append((p, False))
        grads = {id(self): seed}
        for node in reversed(order):
            g = grads.pop(id(node), None)
            if g is None:
                continue
            if node._fn is None:
                node._accumulate(g)
            else:
                for p, pg in zip(node._parents, node._fn(g)):
                    if pg is None or not p.requires_grad:
                        continue
                    pg = _unbroadcast(pg, p.data.shape)
                    k = id(p)
                    grads[k] = pg if k not in grads else grads[k] + pg

    # -- operator
    def __add__(self, o): return add(self, o)
    def __radd__(self, o): return add(o, self)
    def __sub__(self, o): return sub(self, o)
    def __rsub__(self, o): return sub(o, self)
    def __mul__(self, o): return mul(self, o)
    def __rmul__(self, o): return mul(o, self)
    def __truediv__(self, o): return div(self, o)
    def __rtruediv__(self, o): return div(o, self)
    def __neg__(self): return neg(self)
    def __pow__(self, p): return tpow(self, p)
    def __getitem__(self, idx): return getitem(self, idx)


class Param(Tensor):
    def __init__(self, data, name="", fast=False):
        super().__init__(data, requires_grad=True)
        self.name = name
        self.fast = fast
        self.tg = {}

    def _accumulate(self, g):
        t = CTX.task
        self.tg[t] = g.copy() if t not in self.tg else self.tg[t] + g

    def zero_grad(self):
        self.tg = {}


def _c(x):
    return x if isinstance(x, Tensor) else Tensor(x)


def _op(data, parents, fn):
    rg = any(p.requires_grad for p in parents)
    return Tensor(data, rg, tuple(parents) if rg else (), fn if rg else None)


def detach(x):
    return Tensor(_c(x).data)


# ---------------- aritmetika
def add(a, b):
    a, b = _c(a), _c(b)
    return _op(a.data + b.data, (a, b), lambda g: (g, g))


def sub(a, b):
    a, b = _c(a), _c(b)
    return _op(a.data - b.data, (a, b), lambda g: (g, -g))


def mul(a, b):
    a, b = _c(a), _c(b)
    return _op(a.data * b.data, (a, b), lambda g: (g * b.data, g * a.data))


def div(a, b):
    a, b = _c(a), _c(b)
    return _op(a.data / b.data, (a, b), lambda g: (g / b.data, -g * a.data / (b.data ** 2)))


def neg(a):
    a = _c(a)
    return _op(-a.data, (a,), lambda g: (-g,))


def tpow(a, p):
    a = _c(a)
    return _op(a.data ** p, (a,), lambda g: (g * p * a.data ** (p - 1),))


def matmul(a, b):
    a, b = _c(a), _c(b)
    A, B = a.data, b.data
    out = A @ B
    if A.ndim == 1 and B.ndim == 1:
        fn = lambda g: (g * B, g * A)
    elif A.ndim == 1 and B.ndim == 2:
        fn = lambda g: (B @ g, np.outer(A, g))
    elif A.ndim == 2 and B.ndim == 1:
        fn = lambda g: (np.outer(g, B), A.T @ g)
    elif A.ndim == 2 and B.ndim == 2:
        fn = lambda g: (g @ B.T, A.T @ g)
    else:
        raise ValueError("matmul hanya mendukung rank <= 2")
    return _op(out, (a, b), fn)


# ---------------- reduksi / bentuk
def tsum(x, axis=None, keepdims=False):
    x = _c(x)

    def fn(g):
        if axis is not None and not keepdims:
            g = np.expand_dims(g, axis)
        return (np.broadcast_to(g, x.data.shape).copy(),)
    return _op(x.data.sum(axis=axis, keepdims=keepdims), (x,), fn)


def tmean(x, axis=None, keepdims=False):
    x = _c(x)
    n = x.data.size if axis is None else x.data.shape[axis]
    return tsum(x, axis, keepdims) / float(n)


def reshape(x, shape):
    x = _c(x)
    return _op(x.data.reshape(shape), (x,), lambda g: (g.reshape(x.data.shape),))


def transpose(x):
    x = _c(x)
    return _op(x.data.T, (x,), lambda g: (g.T,))


def concat0(xs):
    xs = [_c(x) for x in xs]
    sizes = [x.data.shape[0] for x in xs]

    def fn(g):
        out, i = [], 0
        for s in sizes:
            out.append(g[i:i + s])
            i += s
        return tuple(out)
    return _op(np.concatenate([x.data for x in xs], axis=0), tuple(xs), fn)


def take(E, idx):
    E = _c(E)
    ix = int(idx) if np.isscalar(idx) else np.asarray(idx)

    def fn(g):
        gE = np.zeros_like(E.data)
        np.add.at(gE, ix, g)
        return (gE,)
    return _op(E.data[ix], (E,), fn)


def getitem(x, idx):
    x = _c(x)

    def fn(g):
        gx = np.zeros_like(x.data)
        np.add.at(gx, idx, g)
        return (gx,)
    return _op(x.data[idx], (x,), fn)


# ---------------- aktivasi & fungsi elementwise
def _sig(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def sigmoid(x):
    x = _c(x)
    s = _sig(x.data)
    return _op(s, (x,), lambda g: (g * s * (1 - s),))


def silu(x):
    x = _c(x)
    s = _sig(x.data)
    return _op(x.data * s, (x,), lambda g: (g * (s + x.data * s * (1 - s)),))


def relu(x):
    x = _c(x)
    return _op(np.maximum(x.data, 0.0), (x,), lambda g: (g * (x.data > 0),))


def softplus(x):
    x = _c(x)
    out = np.log1p(np.exp(-np.abs(x.data))) + np.maximum(x.data, 0.0)
    return _op(out, (x,), lambda g: (g * _sig(x.data),))


def texp(x):
    x = _c(x)
    o = np.exp(x.data)
    return _op(o, (x,), lambda g: (g * o,))


def tlog(x):
    x = _c(x)
    return _op(np.log(x.data), (x,), lambda g: (g / x.data,))


def tsqrt(x):
    x = _c(x)
    o = np.sqrt(x.data)
    return _op(o, (x,), lambda g: (g * 0.5 / o,))


# ---------------- normalisasi & loss
def layer_norm(x, eps=1e-5):
    """LayerNorm sumbu terakhir, tanpa affine (sama dengan ln_np di model.py)."""
    x = _c(x)
    xc = x.data - x.data.mean(-1, keepdims=True)
    inv = 1.0 / np.sqrt((xc * xc).mean(-1, keepdims=True) + eps)
    y = xc * inv

    def fn(g):
        return (inv * (g - g.mean(-1, keepdims=True) - y * (g * y).mean(-1, keepdims=True)),)
    return _op(y, (x,), fn)


def l2norm(x, eps=1e-8):
    """x / sqrt(sum x^2 + eps) pada sumbu terakhir (sama dengan l2n_np)."""
    x = _c(x)
    n = np.sqrt((x.data * x.data).sum(-1, keepdims=True) + eps)
    y = x.data / n

    def fn(g):
        return ((g - y * (g * y).sum(-1, keepdims=True)) / n,)
    return _op(y, (x,), fn)


def cross_entropy(logits, target):
    """logits (V,) + target int -> skalar; logits (N,V) + target (N,) -> rata-rata."""
    z = _c(logits)
    x = z.data
    m = x.max(-1, keepdims=True)
    logp = x - (m + np.log(np.exp(x - m).sum(-1, keepdims=True)))
    t = np.asarray(target)
    if x.ndim == 1:
        ti = int(t)

        def fn(g):
            p = np.exp(logp)
            p[ti] -= 1.0
            return (g * p,)
        return _op(-logp[ti], (z,), fn)
    rows = np.arange(x.shape[0])

    def fn2(g):
        p = np.exp(logp)
        p[rows, t] -= 1.0
        return (g * p / x.shape[0],)
    return _op(-logp[rows, t].mean(), (z,), fn2)
