"""Saron — konvolusi kausal (depthwise + pointwise) + SiLU.

Tujuan: memberi Tingkat B fitur n-gram **terpelajar**, yang bisa digeneralisasi pada konteks
jarang — hal yang tidak bisa dilakukan tabel hitung.

Syarat paritas (#2, ARCHITECTURE_V2.md §7): konvolusi dengan **state yang terbawa antar-chunk**
harus sama persis dengan konvolusi atas seluruh deret. Tanpa ini, batas chunk menjadi titik
kebocoran.
"""
import numpy as np


def saron_state_size(conv_k):
    """Banyak byte masukan yang harus dibawa antar-chunk (k-1)."""
    return max(0, int(conv_k) - 1)


def saron_forward(x, wd, bd, wp, bp, state=None, act=True):
    """Konvolusi kausal depthwise + pointwise.

    x: (T, D) · wd: (K, D) tap depthwise · bd: (D,) · wp: (D, Dout) · bp: (Dout,)
    state: (K-1, D) masukan SEBELUM chunk ini (paling baru di baris terakhir), atau None.
    Mengembalikan (y (T,Dout), state_baru (K-1,D)).
    """
    x = np.asarray(x, dtype=np.float64)
    T, D = x.shape
    K = wd.shape[0]
    if state is None:
        state = np.zeros((saron_state_size(K), D), dtype=np.float64)
    buf = np.concatenate([state, x], axis=0)            # (K-1+T, D)
    z = np.zeros((T, D), dtype=np.float64)
    for j in range(K):
        z += wd[j] * buf[K - 1 - j: K - 1 - j + T]
    z += bd
    if act:
        z = z / (1.0 + np.exp(-z))                      # SiLU
    y = z @ wp + bp
    new_state = buf[-(K - 1):].copy() if K > 1 else np.zeros((0, D), dtype=np.float64)
    return y, new_state
