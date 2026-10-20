"""Inti TMT2: satu langkah waktu lapisan SSM + jejak RTRL diagonal, dan tumpukan lapisan.

Pembagian kerja gradien (sesuai flowchart v2):
  * REKURENSI  h_t = a*h_{t-1} + (1-a)*(x B) + e K,  a = sigmoid(x Wa + ba)   -> RTRL manual (numpy).
      Jacobian h_{t-1} -> h_t adalah diag(a), sehingga sensitivitas tiap parameter cukup
      satu matriks berbentuk sama dengan parameternya:  S <- a*S + dh/dtheta_langsung.
      Gradien = g_i * S[j,i], dengan g = dL/dh_t.
  * FEEDFORWARD (baca LN + memori + GLU + adaptor + prediksi) -> autograd, sedalam SATU langkah.
      h_t dijadikan daun (h_leaf); setelah backward, h_leaf.grad = dL/dh_t.
Asumsi keeksakan (dicek di tests/test_core.py): x_t, e_t, dan memori M dianggap eksogen
(stop-grad). Lompatan antar lapisan juga stop-grad (loss JEPA lokal).
"""
import numpy as np
from .autograd import (Tensor, CTX, matmul, layer_norm, l2norm, silu, transpose, texp,
                       _sig)
from .model import ln_np, l2n_np


# ---------------------------------------------------------------- state
class LayerState:
    """State satu lapisan. Array TIDAK pernah dimutasi in-place, jadi salinan boleh berbagi."""
    __slots__ = ("h", "M", "S", "rec", "kp")

    def __init__(self, d, dm):
        self.h = np.zeros(d)
        self.M = np.zeros((dm, dm))
        self.rec = None                                # StepRec yang menghasilkan state ini
        self.kp = np.zeros(d)                          # ln(x_{t-1}): konteks utk kunci hetero
        self.S = {"Wa": np.zeros((d, d)), "B": np.zeros((d, d)),
                  "K": np.zeros((d, d)), "ba": np.zeros(d)}

    def copy(self):
        n = LayerState.__new__(LayerState)
        n.h, n.M, n.S = self.h, self.M, dict(self.S)
        n.rec = getattr(self, "rec", None)
        n.kp = self.kp
        return n

    def decay_traces(self, factor):
        """Truncated RTRL: kecilkan jejak setelah bobot berubah (factor=0 -> reset)."""
        self.S = {k: v * factor for k, v in self.S.items()}


class StackState:
    def __init__(self, model):
        c = model.cfg
        self.layers = [LayerState(c.d, c.d_mem) for _ in model.layers]
        self.prev_p1 = [None] * len(model.layers)   # prediksi horizon-1 (ternormalisasi) langkah lalu

    def copy(self):
        n = StackState.__new__(StackState)
        n.layers = [s.copy() for s in self.layers]
        n.prev_p1 = list(self.prev_p1)
        return n

    def decay_traces(self, factor):
        for s in self.layers:
            s.decay_traces(factor)

    def zero_traces(self):
        """Salinan dengan jejak RTRL nol (awal segmen replay: sensitivitas dipotong)."""
        n = self.copy()
        n.decay_traces(0.0)
        return n


class StepRec:
    """Catatan satu langkah lapisan. Graf & jejak dipertahankan sampai loss-nya terselesaikan
    (bisa terlambat beberapa langkah untuk horizon 4/16)."""

    def __init__(self, layer, cfg, h_leaf, y, S, use_K):
        self.layer, self.cfg, self.h_leaf, self.y, self.S, self.use_K = layer, cfg, h_leaf, y, S, use_K
        self._pred = {}

    def pred(self, model, l, h):
        """Kepala prediksi horizon h dari keluaran lapisan (dibangun malas)."""
        if h not in self._pred:
            self._pred[h] = matmul(self.y, model.P[(l, h)])
        return self._pred[h]


# ---------------------------------------------------------------- feedforward (autograd)
def ff_readout(cfg, L, x, h_leaf, M):
    """h_t -> y_t. x: Tensor (boleh membawa graf encoder). M: array memori (eksogen).

    cfg.mem_query_x: kueri baca memakai ``ln(x_t)`` (token mentah) alih-alih ``ln(h_t)``.
    ``x.data`` dipakai sehingga gradien TIDAK mengalir lewat jalur kueri -- sama seperti kernel
    jalur cepat (``Tensor(x.data)`` memutus graf).
    """
    hn = layer_norm(h_leaf)
    c = matmul(hn, L.Wc)
    if cfg.use_memory:
        qsrc = layer_norm(Tensor(x.data)) if cfg.mem_query_x else hn
        q = l2norm(matmul(qsrc, L.Wq))
        c = c + matmul(matmul(Tensor(M), q), L.Wm)
    g = silu(matmul(c, L.Wg1)) * matmul(c, L.Wg2)
    f = matmul(g, L.Wo)
    if cfg.adapter_rank > 0:
        f = f + matmul(matmul(c, L.Af), L.Bf)
    return x + f


# ---------------------------------------------------------------- satu langkah lapisan
def layer_step(cfg, L, x, st, e=None, surprise=0.0):
    """x: Tensor (d,). st: LayerState (tidak diubah). e: inovasi (d,) atau None.
    Mengembalikan (StepRec, LayerState baru)."""
    xd = x.data
    a = _sig(xd @ L.Wa.data + L.ba.data)
    u = xd @ L.B.data
    ev = e if (e is not None and cfg.use_innovation) else None

    h_prev = st.h
    h = a * h_prev + (1.0 - a) * u
    if ev is not None:
        h = h + ev @ L.K.data

    # --- jejak RTRL (indeks [j, i]: j = masukan parameter, i = unit state)
    c = (h_prev - u) * a * (1.0 - a)                 # dh_i / d(pa_i)
    S = st.S
    Sn = {
        "Wa": a[None, :] * S["Wa"] + xd[:, None] * c[None, :],
        "B":  a[None, :] * S["B"] + xd[:, None] * (1.0 - a)[None, :],
        "K":  a[None, :] * S["K"] + (0.0 if ev is None else ev[:, None]),
        "ba": a * S["ba"] + c,
    }

    # --- baca (graf) memakai memori SEBELUM tulis
    h_leaf = Tensor(h, requires_grad=True)
    y = ff_readout(cfg, L, x, h_leaf, st.M)

    # --- tulis memori delta, hanya saat kejutan (k,v beku -> M eksogen, tanpa RTRL)
    M = st.M
    if cfg.use_memory:
        xn = ln_np(xd)
        # asosiasi AUTO (kunci & nilai dari x_t) atau HETERO (kunci dari konteks sebelumnya)
        src = st.kp if cfg.mem_hetero else xn
        k = l2n_np(src @ L.Wk)
        v = xn @ L.Wv
        beta = cfg.beta_max if cfg.mem_surprise_fixed else \
            cfg.beta_max * min(1.0, max(0.0, float(surprise)))
        if beta > 0.0:
            M = M + beta * np.outer(v - M @ k, k)

    new = LayerState.__new__(LayerState)
    new.h, new.M, new.S = h, M, Sn
    new.kp = ln_np(xd) if cfg.use_memory else st.kp
    new.rec = None
    return StepRec(L, cfg, h_leaf, y, Sn, ev is not None), new


def rtrl_flush(rec, task):
    """Kontraksi g = dL/dh_t dengan jejak S_t -> gradien parameter rekurensi (per-task).
    Panggil setelah setiap backward() yang melewati rec.y. Mengosongkan rec.h_leaf.grad."""
    g = rec.h_leaf.grad
    if g is None:
        return
    L, S = rec.layer, rec.S
    items = [(L.Wa, g[None, :] * S["Wa"]), (L.B, g[None, :] * S["B"]), (L.ba, g * S["ba"])]
    if rec.use_K:
        items.append((L.K, g[None, :] * S["K"]))
    for p, gr in items:
        p.tg[task] = gr if task not in p.tg else p.tg[task] + gr
    rec.h_leaf.grad = None


def backward_task(tip, recs, task, seed=None):
    """backward dari `tip` untuk `task`, lalu RTRL-flush semua rec yang terlibat."""
    prev = CTX.task
    CTX.task = task
    try:
        tip.backward(seed)
        for r in recs:
            rtrl_flush(r, task)
    finally:
        CTX.task = prev


# ---------------------------------------------------------------- tumpukan lapisan
def stack_step(model, x, st, z_star=None):
    """Satu langkah seluruh tumpukan. x: Tensor latent masukan (d,) (layer 0 boleh membawa
    graf encoder). z_star: latent target-encoder untuk byte yang BARU tiba (untuk inovasi).
    Mengembalikan (recs, StackState baru)."""
    cfg = model.cfg
    layers, recs, prev_p1 = [], [], []
    xin = x
    zs = None if z_star is None else l2n_np(z_star)
    for l, L in enumerate(model.layers):
        e, s = None, 0.0
        pp = st.prev_p1[l]
        if cfg.use_innovation and zs is not None and pp is not None:
            diff = zs - pp
            e = np.sqrt(cfg.d) * diff              # skala O(1) per dimensi
            s = 0.5 * float(diff @ diff)           # = 1 - cos (kejutan)
        rec, ns = layer_step(cfg, L, xin, st.layers[l], e, s)
        recs.append(rec)
        layers.append(ns)
        prev_p1.append(l2n_np(rec.pred(model, l, 1).data))
        xin = Tensor(rec.y.data)                   # stop-grad antar lapisan (loss JEPA lokal)
    new = StackState.__new__(StackState)
    new.layers, new.prev_p1 = layers, prev_p1
    return recs, new


# ---------------------------------------------------------------- kepala byte
def byte_logits(model, p, D=None):
    """Kepala byte.
    ``head="linear"``: logits = W_out @ l2norm(p) + byte_bias. Masukan dinormalisasi-L2 supaya
      logits terbatas (pred tak dibatasi -> divergen), tetapi W_out TIDAK dinormalisasi sehingga
      tiap kelas punya skala sendiri (berbeda dari kepala kosinus yang hanya punya satu skala global).
    ``head="cosine"`` (lama): scale * cos(p, dict) + byte_bias dengan satu skala global.
    D = l2norm(model.dict_tensor()) boleh dihitung sekali dan dipakai ulang (hanya kosinus)."""
    if model.cfg.head == "linear":
        return matmul(l2norm(p), transpose(model.W_out)) + model.byte_bias
    if D is None:
        D = l2norm(model.dict_tensor())
    return texp(model.logit_scale) * matmul(l2norm(p), transpose(D)) + model.byte_bias
