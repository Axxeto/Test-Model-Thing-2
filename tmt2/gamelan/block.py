"""Blok Tingkat B: Saron + Gong + gerbang per head + SwiGLU + RMSNorm.

Satu blok, `L` blok ditumpuk. Dua mode yang HARUS identik (tes paritas #3):

  * ``mode="chunk"``     — evaluasi PARALEL: seluruh deret diproses per-chunk (matmul/BLAS).
  * ``mode="recurrent"`` — decode sekuensial: satu langkah demi satu langkah, state dibawa.

Inilah yang membuat Tingkat B bisa *teacher-forced* secara paralel saat penilaian, sementara
generasi tetap O(1) per langkah. Tanpa kesetaraan ini, batas chunk menjadi titik kebocoran.

Keluaran blok = `y` (T, d). Konversi ke probabilitas bit (Tingkat C) belum ada di sini.
"""
import numpy as np

from .gong import gong_chunkwise, gong_recurrent
from .saron import saron_forward, saron_state_size


def rmsnorm(x, g, eps=1e-6):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * g


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class GamelanBlock:
    """Satu blok Tingkat B (acuan numpy float64)."""

    def __init__(self, d=32, heads=4, conv_k=8, seed=0, tau_spread=False,
                 tau_min=2.0, tau_max=4096.0):
        rng = np.random.default_rng(seed)
        self.d, self.h, self.conv_k = int(d), int(heads), int(conv_k)
        self.dh = self.d // self.h
        s = 1.0 / np.sqrt(self.d)
        self.wd = rng.normal(0, s, (self.conv_k, self.d))
        self.bd = np.zeros(self.d)
        self.wp = rng.normal(0, s, (self.d, self.d))
        self.wq = rng.normal(0, s, (self.d, self.h * self.dh))
        self.wk = rng.normal(0, s, (self.d, self.h * self.dh))
        self.wv = rng.normal(0, s, (self.d, self.h * self.dh))
        self.wl = rng.normal(0, s, (self.d, self.h))
        # Sebaran τ (dokumen §4.2: "τ_h per head tersebar geometris, mis. 2…4096").
        # CATATAN TERUKUR 2026-10-05: dengan `bl = 0`, λ_h = sigmoid(0) = 0,5 untuk SEMUA head
        # -> horizon awal τ ≈ 1 langkah. Menyalakan `tau_spread=True` (τ = geomspace(2,4096,H))
        # SEMPAT diuji pada 250k byte Indonesia: BPB tahan **2,7300 vs 2,6654** (bl=0) —
        # jadi sedikit LEBIH BURUK pada skala itu, dan TIDAK menghilangkan keruntuhan generasi.
        # Karena λ tetap bergantung data (`h @ wl`), model BISA belajar memisahkan skala sendiri;
        # bias hanya titik awal. Maka default = False (pemenang terukur); flag disediakan untuk
        # diuji ulang pada skala besar.
        self.bl = np.zeros(self.h)
        if tau_spread and self.h > 1:
            import numpy as _np
            tau = _np.geomspace(float(tau_min), float(tau_max), self.h)
            self.bl = _np.log(_np.maximum(tau - 1.0, 1e-6))
        self.tau_spread = bool(tau_spread)
        self.wo = rng.normal(0, s, (self.h * self.dh, self.d))
        self.wg = rng.normal(0, s, (self.d, self.h))
        self.bg = np.zeros(self.h)
        self.w1 = rng.normal(0, s, (self.d, 2 * self.d))   # gerbang & nilai SwiGLU
        self.w2 = rng.normal(0, s, (self.d, self.d))       # proyeksi keluar (sw sudah (T,d))
        self.g1 = np.ones(self.d)
        self.g2 = np.ones(self.d)

    # ---------------------------------------------------------------- state
    def init_state(self):
        return (np.zeros((saron_state_size(self.conv_k), self.d)),
                np.zeros((self.h, self.dh, self.dh)))

    def _gong(self, q, k, v, lam, U, mode, chunk):
        if mode == "chunk":
            return gong_chunkwise(q, k, v, lam, U, chunk=chunk)
        return gong_recurrent(q, k, v, lam, U)

    # ---------------------------------------------------------------- satu langkah (decode)
    def step(self, x_t, state):
        """Satu langkah waktu. x_t: (d,) ; state dari `init_state`. Mengembalikan (y (d,), state)."""
        x = np.asarray(x_t, dtype=np.float64)[None, :]           # (1, d)
        y, ns = self.forward(x, state, mode="recurrent")
        return y[0], ns

    # ---------------------------------------------------------------- seluruh deret
    def forward(self, x, state=None, mode="chunk", chunk=64):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        T = x.shape[0]
        s_state, U = self.init_state() if state is None else state
        h = rmsnorm(x, self.g1)
        # ---- Saron
        xs, s_state = saron_forward(h, self.wd, self.bd, self.wp, np.zeros(self.d), s_state)
        # ---- proyeksi + peluruhan bergantung-data
        q = (h @ self.wq).reshape(T, self.h, self.dh)
        k = (h @ self.wk).reshape(T, self.h, self.dh)
        v = (h @ self.wv).reshape(T, self.h, self.dh)
        lam = _sigmoid(h @ self.wl + self.bl)                    # (T, H)
        # ---- Gong (chunkwise = paralel, recurrent = sekuensial)
        o, U = self._gong(q, k, v, lam, U, mode, chunk)
        o = o.reshape(T, self.h * self.dh) @ self.wo             # (T, d)
        # ---- gerbang per head antara Saron dan Gong
        gate = _sigmoid(h @ self.wg + self.bg)                   # (T, H)
        mix = (gate[:, :, None] * xs.reshape(T, self.h, self.dh)
               + (1.0 - gate[:, :, None]) * o.reshape(T, self.h, self.dh)).reshape(T, self.d)
        # ---- SwiGLU + residual
        z = mix @ self.w1
        a, b = z[:, :self.d], z[:, self.d:]
        sw = (a / (1.0 + np.exp(-a))) * b
        ff = sw @ self.w2
        y = rmsnorm(x + mix + ff, self.g2)
        return y, (s_state, U)
