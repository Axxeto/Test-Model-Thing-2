"""Tes paritas Tingkat B (`tmt2/gamelan/`) — ARCHITECTURE_V2.md §7 #1-#3.

Yang diuji: dualitas bentuk. Kalau bentuk paralel (chunkwise) dan sekuensial (rekuren/decode)
TIDAK identik, maka batas chunk menjadi titik kebocoran dan seluruh klaim kecepatan Tingkat B
tidak sah. Ini pengulangan tradisi TMT2: acuan eksak sebelum jalur cepat.

CATATAN: tes #4 (gradien vs beda-hingga) BELUM ada — Tingkat B belum punya jalur mundur.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tmt2.gamelan import gong_chunkwise, gong_recurrent, saron_forward  # noqa: E402
from tmt2.gamelan.block import GamelanBlock                            # noqa: E402


def _rng(seed=0):
    return np.random.default_rng(seed)


def test_gong_duality():
    """Chunkwise == rekuren untuk berbagai ukuran chunk dan pola λ (target < 1e-10)."""
    rng = _rng(1)
    T, H, D, Dv = 200, 3, 8, 6
    q = rng.normal(0, 1, (T, H, D))
    k = rng.normal(0, 1, (T, H, D))
    v = rng.normal(0, 1, (T, H, Dv))
    worst = 0.0
    for lam_name, lam in (
        ("λ=0,5 tetap", np.full((T, H), 0.5)),
        ("λ acak uniform", rng.uniform(0.05, 0.99, (T, H))),
        ("λ sangat kecil", rng.uniform(0.01, 0.15, (T, H))),
        ("λ mendekati 1", rng.uniform(0.95, 0.999, (T, H))),
    ):
        U0 = rng.normal(0, 0.3, (H, D, Dv))
        o_ref, U_ref = gong_recurrent(q, k, v, lam, U0)
        for C in (1, 2, 7, 32, 64, 200, 256):
            o_c, U_c = gong_chunkwise(q, k, v, lam, U0, chunk=C)
            e = max(np.abs(o_c - o_ref).max(), np.abs(U_c - U_ref).max())
            worst = max(worst, e)
            assert e < 1e-10, f"{lam_name} C={C}: galat {e:.3e}"
    print(f"  Gong chunkwise == rekuren (galat terburuk {worst:.2e}): OK")


def test_saron_state_parity():
    """Konvolusi dengan state terbawa == konvolusi atas seluruh deret (tanpa kebocoran)."""
    rng = _rng(2)
    T, D, K = 300, 12, 8
    x = rng.normal(0, 1, (T, D))
    wd = rng.normal(0, 0.3, (K, D))
    bd = rng.normal(0, 0.1, D)
    wp = rng.normal(0, 0.3, (D, D))
    bp = rng.normal(0, 0.1, D)
    y_full, _ = saron_forward(x, wd, bd, wp, bp)
    worst = 0.0
    for C in (1, 3, 16, 64, 97):
        st = None
        outs = []
        for s in range(0, T, C):
            y, st = saron_forward(x[s:s + C], wd, bd, wp, bp, state=st)
            outs.append(y)
        y_ch = np.concatenate(outs, axis=0)
        e = np.abs(y_ch - y_full).max()
        worst = max(worst, e)
        assert e < 1e-12, f"C={C}: galat {e:.3e}"
    print(f"  Saron state antar-chunk == deret penuh (galat terburuk {worst:.2e}): OK")


def test_block_parallel_equals_sequential():
    """Blok penuh: evaluasi paralel (chunk) == decode sekuensial satu langkah (< 1e-9)."""
    rng = _rng(3)
    T, d = 180, 32
    blk = GamelanBlock(d=d, heads=4, conv_k=8, seed=7)
    x = rng.normal(0, 1, (T, d))
    for C in (1, 5, 32, 64, 180):
        y_par, st_par = blk.forward(x, mode="chunk", chunk=C)
        st = blk.init_state()
        ys = []
        for t in range(T):
            y_t, st = blk.step(x[t], st)
            ys.append(y_t)
        y_seq = np.asarray(ys)
        e = np.abs(y_par - y_seq).max()
        assert e < 1e-9, f"C={C}: galat {e:.3e}"
    # state akhir juga harus sama
    e_state = np.abs(st_par[1] - st[1]).max()
    assert e_state < 1e-9, f"state akhir beda {e_state:.3e}"
    print(f"  Blok: paralel == decode sekuensial (galat < 1e-9), state akhir sama: OK")


def test_gong_state_is_finite():
    """Kestabilan: ‖U‖ tetap terbatas pada peluruhan panjang (τ besar)."""
    rng = _rng(4)
    T, H, D = 4000, 2, 8
    q = rng.normal(0, 1, (T, H, D))
    k = rng.normal(0, 1, (T, H, D))
    v = rng.normal(0, 1, (T, H, D))
    lam = np.full((T, H), 0.9995)                  # τ ≈ 2000 langkah
    _, U = gong_recurrent(q, k, v, lam, np.zeros((H, D, D)))
    n = float(np.abs(U).max())
    assert np.isfinite(n) and n < 1e4, f"‖U‖ = {n}"
    print(f"  ‖U‖ terbatas pada τ≈2000 ({n:.2f}): OK")


def test_block_gradcheck():
    """#4 gradien Tingkat B vs beda-hingga, lewat TURUNAN BERARAH (target rel < 1e-6).

    Dipakai turunan berarah, bukan per-entri: uji per-entri menabrak entri yang gradiennya
    mendekati nol dan memberi galat relatif palsu ~1. Turunan berarah memakai satu arah acak
    untuk SELURUH tensor, jadi jauh lebih kokoh.
    """
    from tmt2.gamelan.train import block_forward_cache, block_backward
    rng = _rng(5)
    blk = GamelanBlock(d=8, heads=2, conv_k=4, seed=11)
    W_out = rng.normal(0, 0.3, (256, blk.d))
    x = rng.normal(0, 1, (12, blk.d))
    ytar = rng.integers(0, 256, 12)
    n = len(ytar)

    def loss_only():
        y, _ = block_forward_cache(blk, x)
        lo = y @ W_out.T
        lo = lo - lo.max(1, keepdims=True)
        E = np.exp(lo)
        P = E / E.sum(1, keepdims=True)
        return float(-np.mean(np.log(P[np.arange(n), ytar] + 1e-300)))

    y, cache = block_forward_cache(blk, x)
    lo = y @ W_out.T
    lo = lo - lo.max(1, keepdims=True)
    E = np.exp(lo)
    P = E / E.sum(1, keepdims=True)
    dl = P.copy()
    dl[np.arange(n), ytar] -= 1.0
    dl /= n
    _, g = block_backward(blk, cache, dl @ W_out)

    eps = 1e-6
    worst, worst_name = 0.0, ""
    for name in ("wd", "bd", "wp", "wq", "wk", "wv", "wl", "bl", "wo", "wg", "bg",
                 "w1", "w2", "g1", "g2"):
        Pp = getattr(blk, name)
        v = rng.normal(0, 1, Pp.shape)
        v /= np.linalg.norm(v)
        ana = float(np.sum(g[name] * v))
        Pp += eps * v
        lp = loss_only()
        Pp -= 2.0 * eps * v
        lm = loss_only()
        Pp += eps * v
        num = (lp - lm) / (2.0 * eps)
        rel = abs(num - ana) / max(1e-9, abs(num), abs(ana))
        if rel > worst:
            worst, worst_name = rel, name
    assert worst < 1e-6, f"gradcheck gagal: rel terburuk {worst:.3e} pada {worst_name}"
    print(f"  gradcheck Tingkat B vs beda-hingga (rel terburuk {worst:.2e}): OK")


def test_tau_spread_geometric():
    """Bias peluruhan harus menyebar τ_h secara geometris (ARCHITECTURE_V2.md §4.2).

    Versi lama memakai `bl = 0` -> λ_h = 0,5 untuk SEMUA head -> horizon memori ≈ 1 langkah,
    dan generasi Tingkat B runtuh ke tarikan tetap (prompt-independent). Regresi ini menjaga
    agar sebaran τ tetap ada.
    """
    blk = GamelanBlock(d=32, heads=4, conv_k=8, seed=0, tau_spread=True)
    lam0 = 1.0 / (1.0 + np.exp(-blk.bl))
    tau = 1.0 / np.maximum(1.0 - lam0, 1e-12)
    assert tau[0] < 3.0, f"τ head pertama terlalu besar: {tau[0]}"
    assert tau[-1] > 1000.0, f"τ head terakhir terlalu kecil: {tau[-1]}"
    assert np.all(np.diff(tau) > 0), f"τ tidak menaik: {tau}"
    blk2 = GamelanBlock(d=32, heads=4, conv_k=8, seed=0, tau_spread=False)
    assert np.allclose(blk2.bl, 0.0), "tau_spread=False harus memberi bias nol"
    print(f"  Sebaran τ geometris: {np.round(tau, 1)}: OK")


if __name__ == "__main__":
    for fn in (test_gong_duality, test_saron_state_parity,
               test_block_parallel_equals_sequential, test_gong_state_is_finite,
               test_block_gradcheck, test_tau_spread_geometric):
        print(f"[{fn.__name__}]")
        fn()
    print("SEMUA TES PARITAS TINGKAT B LULUS")
