"""Tingkat B — jalur mundur + pelatihan (BPTT di dalam chunk, acuan numpy float64).

Bentuk rekuren dan chunkwise punya GRADIEN yang identik (keduanya matematis sama), jadi jalur
mundur ditulis sekali pada bentuk rekuren — lebih sederhana dan lebih mudah diperiksa — sementara
bentuk chunkwise dipakai untuk maju cepat. Yang wajib: **gradcheck terhadap beda-hingga** (#4).

Gradien di sini eksak terhadap implementasi maju di `block.py`; kalau `block.py` diubah,
gradcheck akan menangkapnya.
"""
import numpy as np

from .block import GamelanBlock, _sigmoid
from .saron import saron_state_size


def _rmsnorm_fwd(x, g, eps=1e-6):
    v = (x * x).mean(-1, keepdims=True) + eps
    s = np.sqrt(v)
    xh = x / s
    return xh * g, (xh, s)


def _rmsnorm_bwd(dy, xh, s, g):
    dxh = dy * g
    dx = (dxh - xh * (dxh * xh).mean(-1, keepdims=True)) / s
    dg = (dy * xh).sum(0)
    return dx, dg


def _silu(z):
    return z / (1.0 + np.exp(-z))


def _silu_grad(z):
    s = _sigmoid(z)
    return s * (1.0 + z * (1.0 - s))


def block_forward_cache(blk, x):
    """Maju satu blok + simpan semua yang dibutuhkan jalur mundur (bentuk REKUREN)."""
    T = x.shape[0]
    h, (xh1, s1) = _rmsnorm_fwd(x, blk.g1)
    # ---- Saron
    K = blk.conv_k
    buf = np.concatenate([np.zeros((K - 1, blk.d)), h], axis=0)
    zc = np.zeros((T, blk.d))
    for j in range(K):
        zc += blk.wd[j] * buf[K - 1 - j: K - 1 - j + T]
    zc += blk.bd
    ac = _silu(zc)
    xs = ac @ blk.wp
    # ---- proyeksi
    q = (h @ blk.wq).reshape(T, blk.h, blk.dh)
    k = (h @ blk.wk).reshape(T, blk.h, blk.dh)
    v = (h @ blk.wv).reshape(T, blk.h, blk.dh)
    lam = _sigmoid(h @ blk.wl + blk.bl)
    # ---- Gong (rekuren) — simpan U_t tiap langkah
    U = np.zeros((blk.h, blk.dh, blk.dh))
    Us = np.empty((T, blk.h, blk.dh, blk.dh))
    o = np.empty((T, blk.h, blk.dh))
    for t in range(T):
        for hh in range(blk.h):
            U[hh] = lam[t, hh] * U[hh] + (1.0 - lam[t, hh]) * np.outer(k[t, hh], v[t, hh])
            Us[t, hh] = U[hh]
            o[t, hh] = q[t, hh] @ U[hh]
    of = o.reshape(T, blk.h * blk.dh)
    og = of @ blk.wo                                  # PENTING: blok memakai og, bukan o mentah
    gate = _sigmoid(h @ blk.wg + blk.bg)
    mix = (gate[:, :, None] * xs.reshape(T, blk.h, blk.dh)
           + (1.0 - gate[:, :, None]) * og.reshape(T, blk.h, blk.dh)).reshape(T, blk.d)
    z1 = mix @ blk.w1
    a, b = z1[:, :blk.d], z1[:, blk.d:]
    sw = _silu(a) * b
    ff = sw @ blk.w2
    u = x + mix + ff
    y, (xh2, s2) = _rmsnorm_fwd(u, blk.g2)
    cache = dict(h=h, xh1=xh1, s1=s1, buf=buf, zc=zc, ac=ac, xs=xs, q=q, k=k, v=v, lam=lam,
                 Us=Us, o=o, of=of, og=og, gate=gate, mix=mix, z1=z1, a=a, b=b, sw=sw, ff=ff,
                 u=u, xh2=xh2, s2=s2, y=y, T=T)
    return y, cache


def block_backward(blk, cache, dy):
    """Mundur satu blok. Mengembalikan (dx, grads dict)."""
    T = cache["T"]
    g = {}
    du, g["g2"] = _rmsnorm_bwd(dy, cache["xh2"], cache["s2"], blk.g2)
    # ff = sw @ w2
    g["w2"] = cache["sw"].T @ du
    dsw = du @ blk.w2.T
    # sw = silu(a) * b
    da = dsw * cache["b"] * _silu_grad(cache["a"])
    db = dsw * _silu(cache["a"])
    dz1 = np.concatenate([da, db], axis=1)
    g["w1"] = cache["mix"].T @ dz1
    dmix = dz1 @ blk.w1.T
    dx = du.copy()                                   # residual x
    dmix_total = dmix + du
    dm = dmix_total.reshape(T, blk.h, blk.dh)
    og_h = cache["og"].reshape(T, blk.h, blk.dh)      # mix memakai og, BUKAN o mentah
    dgate = (dm * (cache["xs"].reshape(T, blk.h, blk.dh) - og_h)).sum(-1)
    dxs = (dm * cache["gate"][:, :, None]).reshape(T, blk.d)
    do = dm * (1.0 - cache["gate"])[:, :, None]
    # gate = sigmoid(h@wg+bg)
    dzg = dgate * cache["gate"] * (1.0 - cache["gate"])
    g["wg"] = cache["h"].T @ dzg
    g["bg"] = dzg.sum(0)
    dh = dzg @ blk.wg.T
    # o = of @ wo  (gradien harus lewat wo, kalau tidak jalur Gong terputus dari loss)
    dof = do.reshape(T, blk.h * blk.dh)
    g["wo"] = cache["of"].T @ dof
    do_h = (dof @ blk.wo.T).reshape(T, blk.h, blk.dh)
    # ---- Gong mundur (rekuren)
    dq = np.zeros_like(cache["q"])
    dk = np.zeros_like(cache["k"])
    dv = np.zeros_like(cache["v"])
    dlam = np.zeros_like(cache["lam"])
    dU = np.zeros((blk.h, blk.dh, blk.dh))
    for t in range(T - 1, -1, -1):
        for hh in range(blk.h):
            Ucur = cache["Us"][t, hh]
            dUt = dU[hh] + np.outer(cache["q"][t, hh], do_h[t, hh])
            dq[t, hh] = Ucur @ do_h[t, hh]
            Uprev = cache["Us"][t - 1, hh] if t > 0 else np.zeros((blk.dh, blk.dh))
            l = cache["lam"][t, hh]
            dlam[t, hh] = float(np.sum(dUt * (Uprev - np.outer(cache["k"][t, hh],
                                                               cache["v"][t, hh]))))
            dU[hh] = l * dUt
            dk[t, hh] = (1.0 - l) * (dUt @ cache["v"][t, hh])
            dv[t, hh] = (1.0 - l) * (dUt.T @ cache["k"][t, hh])
    # lam = sigmoid(h@wl+bl)
    dzl = dlam * cache["lam"] * (1.0 - cache["lam"])
    g["wl"] = cache["h"].T @ dzl
    g["bl"] = dzl.sum(0)
    dh += dzl @ blk.wl.T
    # q,k,v = h @ w
    dqf, dkf, dvf = dq.reshape(T, -1), dk.reshape(T, -1), dv.reshape(T, -1)
    g["wq"] = cache["h"].T @ dqf
    g["wk"] = cache["h"].T @ dkf
    g["wv"] = cache["h"].T @ dvf
    dh += dqf @ blk.wq.T + dkf @ blk.wk.T + dvf @ blk.wv.T
    # ---- Saron mundur
    K = blk.conv_k
    g["wp"] = cache["ac"].T @ dxs
    dac = dxs @ blk.wp.T
    dzc = dac * _silu_grad(cache["zc"])
    g["bd"] = dzc.sum(0)
    dbuf = np.zeros((K - 1 + T, blk.d))
    g["wd"] = np.zeros((K, blk.d))
    for j in range(K):
        sl = slice(K - 1 - j, K - 1 - j + T)
        g["wd"][j] = (dzc * cache["buf"][sl]).sum(0)
        dbuf[sl] += dzc * blk.wd[j]
    dh += dbuf[K - 1:]
    # h = rmsnorm(x, g1)
    dx_h, g["g1"] = _rmsnorm_bwd(dh, cache["xh1"], cache["s1"], blk.g1)
    dx = dx + dx_h
    return dx, g
