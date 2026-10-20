"""Gradcheck beda-hingga (central difference) untuk setiap operator autograd + perilaku graf."""
import numpy as np
from tmt2.autograd import *
from tmt2.autograd import CTX, _unbroadcast

RNG = np.random.default_rng(0)


def numgrad(f, x, eps=1e-6):
    g = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        a, b = x.copy(), x.copy()
        a[i] += eps
        b[i] -= eps
        g[i] = (f(a) - f(b)) / (2 * eps)
    return g


def check(fn, *arrs, tol=1e-6, name=""):
    """fn(*Tensor) -> Tensor; skalarkan dengan bobot acak tetap lalu bandingkan dengan beda-hingga."""
    ts = [Tensor(a, requires_grad=True) for a in arrs]
    out = fn(*ts)
    w = RNG.normal(size=out.data.shape)
    loss = tsum(out * w) if out.data.ndim else out * float(w)
    loss.backward()
    for k, a in enumerate(arrs):
        def f(x, k=k):
            args = [Tensor(b) for b in arrs]
            args[k] = Tensor(x)
            o = fn(*args)
            return float(np.sum(o.data * w))
        ng = numgrad(f, a.astype(float))
        err = np.abs(ts[k].grad - ng).max() / (np.abs(ng).max() + 1e-9)
        assert err < tol, (name, k, err)


def test_arithmetic_and_broadcast():
    a, b = RNG.normal(size=(3, 4)), RNG.normal(size=(4,))
    check(lambda x, y: x + y, a, b, name="add-bcast")
    check(lambda x, y: x - y, a, b, name="sub-bcast")
    check(lambda x, y: x * y, a, b, name="mul-bcast")
    check(lambda x, y: x / (y * y + 1.0), a, b, name="div")
    check(lambda x: -x, a, name="neg")
    check(lambda x: x ** 3, np.abs(a) + 0.5, name="pow")
    check(lambda x: 2.0 - x * 3.0 + 1.0 / (x * x + 1.0), a, name="scalar-mix")
    c = RNG.normal(size=(3, 1))
    check(lambda x, y: x * y, a, c, name="mul-col-bcast")


def test_matmul_all_ranks():
    A, B = RNG.normal(size=(3, 4)), RNG.normal(size=(4, 5))
    v, u = RNG.normal(size=4), RNG.normal(size=4)
    check(matmul, A, B, name="mm22")
    check(matmul, v, B, name="mm12")
    check(matmul, A, v, name="mm21")
    check(matmul, v, u, name="mm11")


def test_reductions_and_shapes():
    a = RNG.normal(size=(3, 4))
    check(lambda x: tsum(x), a, name="sum")
    check(lambda x: tsum(x, axis=0), a, name="sum0")
    check(lambda x: tsum(x, axis=1, keepdims=True), a, name="sum1k")
    check(lambda x: tmean(x, axis=0), a, name="mean0")
    check(lambda x: tmean(x), a, name="mean")
    check(lambda x: reshape(x, (4, 3)), a, name="reshape")
    check(lambda x: transpose(x), a, name="T")
    check(lambda x, y: concat0([x, y]), a, RNG.normal(size=(2, 4)), name="concat0")
    check(lambda x: take(x, 2), a, name="take-int")
    check(lambda x: take(x, np.array([0, 2, 2])), a, name="take-repeat")
    check(lambda x: x[1], a, name="getitem")


def test_elementwise():
    a = RNG.normal(size=(3, 4)) * 2
    p = np.abs(a) + 0.3
    check(sigmoid, a, name="sigmoid")
    check(silu, a, name="silu")
    check(softplus, a, name="softplus")
    check(texp, a * 0.5, name="exp")
    check(tlog, p, name="log")
    check(tsqrt, p, name="sqrt")
    a2 = a + 0.05 * np.sign(a)          # jauhi kink relu di 0
    check(relu, a2, name="relu")


def test_norm_and_loss():
    a = RNG.normal(size=(3, 6))
    check(layer_norm, a, name="layer_norm")
    check(l2norm, a, name="l2norm")
    check(layer_norm, a[0], name="layer_norm-1d")
    check(l2norm, a[0], name="l2norm-1d")
    tgt = np.array([1, 5, 0])
    check(lambda z: cross_entropy(z, tgt), a, name="ce-batch")
    check(lambda z: cross_entropy(z, 3), a[0], name="ce-1d")
    # nilai CE = -log softmax
    z = a[0]
    ref = -(z[3] - np.log(np.exp(z).sum()))
    assert abs(float(cross_entropy(Tensor(z), 3).data) - ref) < 1e-12


def test_unbroadcast():
    g = np.ones((3, 4))
    assert _unbroadcast(g, (4,)).shape == (4,) and (_unbroadcast(g, (4,)) == 3).all()
    assert _unbroadcast(g, (3, 1)).shape == (3, 1) and (_unbroadcast(g, (3, 1)) == 4).all()
    assert _unbroadcast(g, ()).shape == () and float(_unbroadcast(g, ())) == 12


def test_graph_reuse_and_accumulation():
    """Backward berkali-kali pada graf yang sama; daun biasa mengakumulasi, Param per-task."""
    x = Tensor(RNG.normal(size=3), requires_grad=True)
    y = tsum(x * x)
    y.backward(); y.backward()
    assert np.allclose(x.grad, 4 * x.data)
    p = Param(RNG.normal(size=3), "p")
    q = tsum(p * p)
    CTX.task = "a"; q.backward()
    CTX.task = "b"; q.backward(); q.backward()
    CTX.task = "default"
    assert np.allclose(p.tg["a"], 2 * p.data) and np.allclose(p.tg["b"], 4 * p.data)
    p.zero_grad()
    assert p.tg == {}


def test_no_grad_paths():
    c = Tensor(np.ones(3))                       # bukan requires_grad -> tanpa graf
    z = c * 2.0 + 1.0
    assert not z.requires_grad and z._parents == ()
    z.backward()                                 # no-op, tidak error
    x = Tensor(np.ones(3), requires_grad=True)
    assert not detach(x * 3).requires_grad
    # diamond: gradien dari dua jalur dijumlahkan
    y = x * 2.0
    (y * y + y).backward(seed=np.ones(3))
    assert np.allclose(x.grad, 8 * x.data + 2)


def test_deep_graph_no_recursion_limit():
    x = Tensor(np.array(1.0), requires_grad=True)
    y = x
    for _ in range(5000):
        y = y * 1.0001
    y.backward()
    assert abs(float(x.grad) - 1.0001 ** 5000) < 1e-6


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name); fn()
    print("autograd ok")
