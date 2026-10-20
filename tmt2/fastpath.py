"""Jalur panas TMT2 dalam numba ``@njit`` (numpy float64).

``tmt2/autograd.py`` tetap menjadi **definisi kebenaran** dan dipakai oleh tes/validasi.
Modul ini punya dua lapis:

  A. Kernel SATUAN (dipakai tes + sebagai acuan silang)
       rec_fwd / rtrl_flush        rekurensi + jejak RTRL diagonal
       readout_fwd / readout_bwd   LN + memori + GLU + adaptor, maju & mundur analitik
       head_fwd / head_bwd         kepala byte + CE (termasuk grad kamus E)
       jepa_bwd                    loss JEPA lokal + mundur
       stop_fwd / stop_grads       kepala stop (probe y DETACH, sama seperti learner)
  B. Kernel TERPADU (dipakai loop per-byte, loop eksplisit + scratch buffer)
       step_fwd           rekurensi + RTRL + readout dalam satu lintasan
       flush_rec          backward readout + kontraksi RTRL, gradien langsung ke arena
       state_cap_scatter  penalti batas state -> gradien parameter rekurensi

CATATAN implementasi: ``A @ B`` (``np.dot`` 2-D) di numba memerlukan scipy (BLAS), jadi
semua perkalian matriks ditulis sebagai broadcast+sum atau loop eksplisit. Setiap operasi
array numba berbiaya ~1.5 us, sehingga kernel terpadu menulis ke buffer scratch yang
disediakan pemanggil (tanpa alokasi per operasi) -- ini sumber percepatan utama.

Semua penurunan gradien diturunkan tangan dan diverifikasi terhadap autograd di
``tests/test_fastpath.py``. Jangan ubah tanpa menjalankan tes itu.
"""
import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:                                     # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not (len(a) == 1 and callable(a[0])) else a[0]

_EPS_LN = 1e-5
_EPS_L2 = 1e-8


# ------------------------------------------------------------------ dasar
@njit(cache=True)
def _sig(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


@njit(cache=True)
def _ln(h):
    """LayerNorm tanpa affine (identik dengan ln_np / autograd.layer_norm)."""
    xc = h - h.mean()
    inv = 1.0 / np.sqrt((xc * xc).mean() + _EPS_LN)
    return xc * inv, inv


@njit(cache=True)
def _l2(v):
    n = np.sqrt((v * v).sum() + _EPS_L2)
    return v / n, n


@njit(cache=True, inline="always", fastmath=True)
def _mv(A, v):
    """(m,n) @ (n,) -> (m,)  [tanpa BLAS].

    Loop eksplisit: bentuk broadcast ``(A * v[None,:]).sum(1)`` mengalokasikan dan menulis
    temporer (m,n) penuh setiap panggilan (mis. 256x32 = 64 KB untuk kepala keluaran) lalu
    mereduksinya -- dua lintasan penuh. Loop ini satu lintasan tanpa alokasi: terukur 7,2x
    lebih cepat pada (256,32). Beda numerik vs broadcast ~3e-15 (orde epsilon float64).
    """
    m, n = A.shape
    out = np.zeros(m)
    for i in range(m):
        s = 0.0
        for j in range(n):
            s += A[i, j] * v[j]
        out[i] = s
    return out


@njit(cache=True, inline="always", fastmath=True)
def _vm(v, A):
    """(n,) @ (n,m) -> (m,)  [tanpa BLAS]. Loop luar atas baris -> akses A kontigu."""
    n, m = A.shape
    out = np.zeros(m)
    for i in range(n):
        vi = v[i]
        row = A[i]
        for j in range(m):
            out[j] += vi * row[j]
    return out


@njit(cache=True, inline="always", fastmath=True)
def _dot(u, v):
    s = 0.0
    for i in range(u.shape[0]):
        s += u[i] * v[i]
    return s


# ------------------------------------------------------------------ rekurensi + RTRL
@njit(cache=True)
def rec_fwd(x, hp, S_Wa, S_B, S_K, S_ba, Wa, ba, B, K, e, use_e):
    """Satu langkah rekurensi + pembaruan jejak RTRL. Mengembalikan (h, S_Wa, S_B, S_K, S_ba)."""
    a = _sig(_vm(x, Wa) + ba)
    u = _vm(x, B)
    h = a * hp + (1.0 - a) * u
    if use_e:
        h = h + _vm(e, K)
    c = (hp - u) * a * (1.0 - a)
    nWa = np.expand_dims(a, 0) * S_Wa + np.expand_dims(x, 1) * np.expand_dims(c, 0)
    nB = np.expand_dims(a, 0) * S_B + np.expand_dims(x, 1) * np.expand_dims(1.0 - a, 0)
    nK = np.expand_dims(a, 0) * S_K
    if use_e:
        nK = nK + np.expand_dims(e, 1)
    nba = a * S_ba + c
    return h, nWa, nB, nK, nba


@njit(cache=True)
def rtrl_flush(g, S_Wa, S_B, S_K, S_ba, use_K):
    """grad parameter rekurensi dari g = dL/dh_t (lihat core.rtrl_flush)."""
    gWa = np.expand_dims(g, 0) * S_Wa
    gB = np.expand_dims(g, 0) * S_B
    gba = g * S_ba
    gK = np.zeros(S_K.shape)
    if use_K:
        gK = np.expand_dims(g, 0) * S_K
    return gWa, gB, gba, gK


# ------------------------------------------------------------------ readout
@njit(cache=True)
def readout_fwd(h, x, M, Wc, Wq, Wm, Wg1, Wg2, Wo, Af, Bf, use_mem, use_ad, query_x=False):
    hn, _ = _ln(h)
    c0 = _vm(hn, Wc)
    cf = c0
    if use_mem:
        if query_x:                                   # kueri dari token mentah ln(x)
            qsrc, _ = _ln(x)
        else:                                         # kueri dari state ln(h)
            qsrc = hn
        q, _ = _l2(_vm(qsrc, Wq))
        cf = c0 + _vm(_mv(M, q), Wm)
    z1 = _vm(cf, Wg1)
    s1 = _sig(z1)
    gv = (z1 * s1) * _vm(cf, Wg2)
    f = _vm(gv, Wo)
    if use_ad:
        f = f + _vm(_vm(cf, Af), Bf)
    return x + f


@njit(cache=True)
def readout_bwd(h, x, M, gy, Wc, Wq, Wm, Wg1, Wg2, Wo, Af, Bf, use_mem, use_ad,
                query_x=False):
    """Backward analitik readout. Mengembalikan (g_h, g_x, gWc, gWq, gWm, gWg1, gWg2, gWo, gAf, gBf).

    query_x=True: kueri dibaca dari ``ln(x)``; ``x`` eksogen sehingga gradien kueri tidak
    mengalir ke ``h`` (g_hn tidak menerima ``Wq @ g_u``).
    """
    d = h.shape[0]
    dm = M.shape[0]
    r = Af.shape[1]
    hn, inv = _ln(h)
    c0 = _vm(hn, Wc)
    if use_mem:
        if query_x:
            qsrc, _ = _ln(x)
        else:
            qsrc = hn
        q, nq = _l2(_vm(qsrc, Wq))
        am = _mv(M, q)
        cf = c0 + _vm(am, Wm)
    else:
        q = np.zeros(dm)
        nq = 1.0
        am = np.zeros(dm)
        cf = c0
    z1 = _vm(cf, Wg1)
    s1 = _sig(z1)
    z2 = _vm(cf, Wg2)
    gv = (z1 * s1) * z2
    g_gv = _mv(Wo, gy)
    gWo = np.outer(gv, gy)
    g_cf = np.zeros(d)
    gAf = np.zeros((d, r))
    gBf = np.zeros((r, d))
    if use_ad:
        b1 = _vm(cf, Af)
        g_b1 = _mv(Bf, gy)
        gBf = np.outer(b1, gy)
        gAf = np.outer(cf, g_b1)
        g_cf += _mv(Af, g_b1)
    g_z1 = g_gv * z2 * (s1 + z1 * s1 * (1.0 - s1))
    g_z2 = g_gv * (z1 * s1)
    gWg1 = np.outer(cf, g_z1)
    gWg2 = np.outer(cf, g_z2)
    g_cf += _mv(Wg1, g_z1) + _mv(Wg2, g_z2)
    gWm = np.zeros((dm, d))
    gWq = np.zeros((d, dm))
    g_hn = np.zeros(d)
    if use_mem:
        g_am = _mv(Wm, g_cf)
        gWm = np.outer(am, g_cf)
        g_q = _vm(g_am, M)                            # M^T @ g_am
        g_u = (g_q - q * _dot(g_q, q)) / nq
        gWq = np.outer(qsrc, g_u)
        if not query_x:                               # pada mode x, x eksogen -> tak ke h
            g_hn = _mv(Wq, g_u)
    gWc = np.outer(hn, g_cf)
    g_hn = g_hn + _mv(Wc, g_cf)
    g_h = inv * (g_hn - g_hn.mean() - hn * (g_hn * hn).mean())
    return g_h, gy, gWc, gWq, gWm, gWg1, gWg2, gWo, gAf, gBf


# ------------------------------------------------------------------ kepala byte
@njit(cache=True)
def head_fwd(p, D, logit_scale, byte_bias):
    s, _ = _l2(p)
    return np.exp(logit_scale) * _mv(D, s) + byte_bias


@njit(cache=True)
def head_bwd(p, D, E, logit_scale, glo, w, P, y):
    """glo = softmax(logits) dengan 1 dikurangi di target; w = bobot task.
    Mengembalikan (g_y, gP, gE, g_logit_scale, g_byte_bias)."""
    s, np_ = _l2(p)
    sc = np.exp(logit_scale)
    g = w * glo
    base = _mv(D, s)
    g_byte_bias = g
    g_logit_scale = sc * _dot(g, base)
    gs = sc * g                                        # dL/d(s@D^T) = sc * dL/dlogits
    g_s = _vm(gs, D)
    g_p = (g_s - s * _dot(g_s, s)) / np_
    g_y = _mv(P, g_p)
    gP = np.outer(y, g_p)
    g_D = np.outer(gs, s)
    n_E = np.sqrt((E * E).sum(axis=1) + _EPS_L2)
    gE = (g_D - D * np.expand_dims((g_D * D).sum(axis=1), 1)) / np.expand_dims(n_E, 1)
    return g_y, gP, gE, g_logit_scale, g_byte_bias


@njit(cache=True, fastmath=True)
def head_bwd_lin_acc(p, W_out, glo, w, P, y, g, oW):
    """Varian hemat dari ``head_bwd_lin``: gradien W_out diakumulasi LANGSUNG ke arena datar
    ``g`` pada offset ``oW``, sehingga tidak ada alokasi (V,d) per token (V=4096 -> 1 MB/token).
    Mengembalikan (g_y, gP, g_byte_bias)."""
    s, np_ = _l2(p)
    V = glo.shape[0]
    d = s.shape[0]
    gv = w * glo
    for i in range(V):
        gi = gv[i]
        base = oW + i * d
        for j in range(d):
            g[base + j] += gi * s[j]
    g_s = np.zeros(d)
    for i in range(V):
        gi = gv[i]
        for j in range(d):
            g_s[j] += gi * W_out[i, j]
    dg = 0.0
    for j in range(d):
        dg += g_s[j] * s[j]
    g_p = np.empty(d)
    for j in range(d):
        g_p[j] = (g_s[j] - s[j] * dg) / np_
    g_y = _mv(P, g_p)
    gP = np.outer(y, g_p)
    return g_y, gP, gv


@njit(cache=True)
def softmax_grad_target(logits, b, w):
    """gradien CE terhadap logits: w * (softmax(logits) - onehot(b))."""
    m = logits.max()
    p = np.exp(logits - m)
    p = p / p.sum()
    p[b] -= 1.0
    return w * p


@njit(cache=True)
def ce_val(logits, b):
    m = logits.max()
    return float(m + np.log(np.exp(logits - m).sum()) - logits[b])


# ------------------------------------------------------------------ kepala byte LINEAR
# logits = W_out @ l2norm(p) + byte_bias. Masukan dinormalisasi-L2 (batas numerik: pred mentah
# tak dibatasi sehingga logits meledak), tetapi W_out tidak -> skala per-kelas tetap bebas.
@njit(cache=True, fastmath=True)
def head_fwd_lin(p, W_out, byte_bias):
    s, _ = _l2(p)
    return _mv(W_out, s) + byte_bias


@njit(cache=True, fastmath=True)
def head_bwd_lin(p, W_out, glo, w, P, y):
    """glo = softmax(logits) dengan 1 dikurangi di target; w = bobot task.
    Mengembalikan (g_y, gP, g_W_out, g_byte_bias)."""
    s, np_ = _l2(p)
    g = w * glo
    g_W = np.outer(g, s)                               # (256, d)
    g_s = _vm(g, W_out)                                # (d,)
    g_p = (g_s - s * _dot(g_s, s)) / np_               # mundur l2norm
    g_y = _mv(P, g_p)
    gP = np.outer(y, g_p)
    return g_y, gP, g_W, g


# ------------------------------------------------------------------ JEPA
@njit(cache=True)
def jepa_bwd(pred, tgt, P, y):
    """0.5*||l2n(pred) - tgt||^2 ; tgt konstan. -> (loss, g_y, gP)."""
    u, npd = _l2(pred)
    diff = u - tgt
    loss = 0.5 * float(_dot(diff, diff))
    g_pred = (diff - u * _dot(diff, u)) / npd
    g_y = _mv(P, g_pred)
    gP = np.outer(y, g_pred)
    return loss, g_y, gP


# ------------------------------------------------------------------ kepala stop
@njit(cache=True)
def stop_fwd(y, ws, bs):
    return _sig(_dot(y, ws) + bs)


@njit(cache=True)
def stop_grads(sp, ces, think_cost, stop_kl, rho):
    """gradien loss stop terhadap sp_k (lihat learner._resolve). Mengembalikan array (n,)."""
    n = sp.shape[0]
    g = np.zeros(n)
    if n < 2:
        return g
    for k in range(n - 1):
        lab = 1.0 if (ces[k] - ces[k + 1]) < think_cost else 0.0
        s = sp[k]
        g[k] += -(lab / (s + 1e-9)) + (1.0 - lab) / (1.0 - s + 1e-9)
    pk = np.zeros(n)
    qk = np.zeros(n)
    rem = 1.0
    for k in range(n):
        if k < n - 1:
            pk[k] = rem * sp[k]
            rem = rem * (1.0 - sp[k])
            qk[k] = rho * (1.0 - rho) ** k
        else:
            pk[k] = rem
            qk[k] = (1.0 - rho) ** k
    dk = np.zeros(n)
    for k in range(n):
        dk[k] = np.log(pk[k] + 1e-9) + pk[k] / (pk[k] + 1e-9) - np.log(qk[k])
    for j in range(n - 1):
        rest = 0.0
        for k in range(j + 1, n):
            rest += dk[k] * pk[k]
        g[j] += stop_kl * (dk[j] * pk[j] / (sp[j] + 1e-9) - rest / (1.0 - sp[j] + 1e-9))
    return g


@njit(cache=True)
def stop_loss(sp, ces, think_cost, stop_kl, rho):
    n = sp.shape[0]
    if n < 2:
        return 0.0
    tot = 0.0
    for k in range(n - 1):
        lab = 1.0 if (ces[k] - ces[k + 1]) < think_cost else 0.0
        s = sp[k]
        tot += -(lab * np.log(s + 1e-9) + (1.0 - lab) * np.log(1.0 - s + 1e-9))
    rem = 1.0
    kl = 0.0
    for k in range(n):
        if k < n - 1:
            pkk = rem * sp[k]
            rem = rem * (1.0 - sp[k])
            qkk = rho * (1.0 - rho) ** k
        else:
            pkk = rem
            qkk = (1.0 - rho) ** k
        kl += pkk * (np.log(pkk + 1e-9) - np.log(qkk))
    return tot + stop_kl * kl


# =====================================================================================
# Kernel TERPADU (loop eksplisit + scratch buffer) untuk loop per-byte.
# =====================================================================================
def scratch_size(d, dm, ed, r):
    """Ukuran scratch untuk step_fwd: 8d + 3dm + 3ed + r (ekor: d untuk ln(x) bila mem_query_x)."""
    return 8 * d + 3 * dm + 3 * ed + r


def scratch_size_bwd(d, dm, ed, r):
    """Ukuran scratch untuk fused_bwd: 6d + 4dm + 6ed + 2r + d(lnx) + 8d + V + 3d (lihat pemakaian)."""
    return 6 * d + 4 * dm + 6 * ed + 2 * r


def scratch_size_mw(d, dm):
    """Ukuran scratch untuk mem_write_bwd: 7d + 6dm."""
    return 7 * d + 6 * dm


@njit(cache=True, fastmath=True)
def mem_write_bwd(x, M, Wk, Wv, beta, gam, q, g, oWk, oWv, ox, scr):
    """Backward penulisan memori (terpangkas satu langkah) -> gradien Wk, Wv, dan g_x.

    ``gM = outer(gam, q)`` adalah dL/dM dari PEMBACAAN pada langkah berikutnya (M_t dihasilkan
    oleh penulisan pada langkah t-1). Karena ``Mk = M k`` dan ``dd = v - Mk``, hanya dua skalar
    yang dibutuhkan:
        g_d = beta * gam * (q . k)      g_k = beta * q * (gam . dd)
    """
    d = x.shape[0]
    dm = M.shape[0]
    hnx = scr[0:d]
    uk = scr[d:d + dm]
    k = scr[d + dm:d + 2 * dm]
    v = scr[d + 2 * dm:d + 3 * dm]
    dd = scr[d + 3 * dm:d + 4 * dm]
    gk = scr[d + 4 * dm:d + 5 * dm]
    gd = scr[d + 5 * dm:d + 6 * dm]
    ghnx = scr[d + 6 * dm:d + 7 * dm]
    mx = 0.0
    for i in range(d):
        mx += x[i]
    mx /= d
    vx = 0.0
    for i in range(d):
        t = x[i] - mx
        vx += t * t
    inv = 1.0 / np.sqrt(vx / d + _EPS_LN)
    for i in range(d):
        hnx[i] = (x[i] - mx) * inv
    for j in range(dm):
        s = 0.0
        for i in range(d):
            s += hnx[i] * Wk[i, j]
        uk[j] = s
    nk = np.sqrt((uk * uk).sum() + _EPS_L2)
    for j in range(dm):
        k[j] = uk[j] / nk
    for j in range(dm):
        s = 0.0
        for i in range(d):
            s += hnx[i] * Wv[i, j]
        v[j] = s
    for i in range(dm):
        s = 0.0
        for j in range(dm):
            s += M[i, j] * k[j]
        dd[i] = v[i] - s
    qk = 0.0
    for j in range(dm):
        qk += q[j] * k[j]
    gdd = 0.0
    for i in range(dm):
        gdd += gam[i] * dd[i]
    for i in range(dm):
        gd[i] = beta * gam[i] * qk
    for j in range(dm):
        # dd = v - M k juga bergantung pada k -> ada suku -M^T g_d
        s = 0.0
        for i in range(dm):
            s += M[i, j] * gd[i]
        gk[j] = beta * q[j] * gdd - s
    dk = 0.0
    for j in range(dm):
        dk += gk[j] * k[j]
    for j in range(dm):
        gk[j] = (gk[j] - k[j] * dk) / nk               # k = l2n(uk): backward l2norm
    for i in range(d):
        hi = hnx[i]
        for j in range(dm):
            g[oWk + i * dm + j] += hi * gk[j]
            g[oWv + i * dm + j] += hi * gd[j]
    for i in range(d):
        s = 0.0
        for j in range(dm):
            s += Wk[i, j] * gk[j] + Wv[i, j] * gd[j]
        ghnx[i] = s
    mgh = 0.0
    for i in range(d):
        mgh += ghnx[i]
    mgh /= d
    dgh = 0.0
    for i in range(d):
        dgh += ghnx[i] * hnx[i]
    dgh /= d
    if ox >= 0:
        for i in range(d):
            g[ox + i] += inv * (ghnx[i] - mgh - hnx[i] * dgh)


@njit(cache=True, fastmath=True)
def step_fwd(x, hp, SWa, SB, SK, Sba, Wa, ba, B, K, e, use_e,
             M, Wc, Wq, Wm, Wg1, Wg2, Wo, Af, Bf, use_mem, use_ad,
             Wk, Wv, kp_prev, surprise, beta_max, hetero, fix_beta, query_x,
             o_h, o_SWa, o_SB, o_SK, o_Sba, o_y, o_M, o_kp, o_a, o_cc, scr):
    """Satu langkah lapisan TERPADU: rekurensi + jejak RTRL + readout + tulis memori.

    o_M = memori setelah tulis (M masukan adalah memori SEBELUM tulis, dipakai readout).
    """
    d = x.shape[0]
    dm = M.shape[0]
    ed = Wg1.shape[1]
    r = Af.shape[1]
    a = scr[0:d]
    u = scr[d:2 * d]
    cc = scr[2 * d:3 * d]
    hn = scr[3 * d:4 * d]
    c0 = scr[4 * d:5 * d]
    cf = scr[5 * d:6 * d]
    ff = scr[6 * d:7 * d]
    uq = scr[7 * d:7 * d + dm]
    q = scr[7 * d + dm:7 * d + 2 * dm]
    am = scr[7 * d + 2 * dm:7 * d + 3 * dm]
    z1 = scr[7 * d + 3 * dm:7 * d + 3 * dm + ed]
    z2 = scr[7 * d + 3 * dm + ed:7 * d + 3 * dm + 2 * ed]
    gv = scr[7 * d + 3 * dm + 2 * ed:7 * d + 3 * dm + 3 * ed]
    b1 = scr[7 * d + 3 * dm + 3 * ed:7 * d + 3 * dm + 3 * ed + r]
    lnx = scr[7 * d + 3 * dm + 3 * ed + r:8 * d + 3 * dm + 3 * ed + r]
    # ---- rekurensi + jejak RTRL
    for i in range(d):
        s = ba[i]
        for j in range(d):
            s += x[j] * Wa[j, i]
        ai = 0.5 * (1.0 + np.tanh(0.5 * s))
        a[i] = ai
        t = 0.0
        for j in range(d):
            t += x[j] * B[j, i]
        u[i] = t
        cc[i] = (hp[i] - t) * ai * (1.0 - ai)
    for i in range(d):
        o_h[i] = a[i] * hp[i] + (1.0 - a[i]) * u[i]
        # keluarkan a dan cc: dipakai flush_rec untuk menghitung d h_i / d x_j.
        # x_l menggerakkan GERBANG a dan u, bukan hanya jalur residual y = x + f;
        # tanpa ini gradien antar-lapisan kehilangan suku itu (lihat catatan deep_grad).
        o_a[i] = a[i]
        o_cc[i] = cc[i]
    if use_e:
        for i in range(d):
            s = 0.0
            for j in range(d):
                s += e[j] * K[j, i]
            o_h[i] += s
    for j in range(d):
        for i in range(d):
            o_SWa[j, i] = a[i] * SWa[j, i] + x[j] * cc[i]
            o_SB[j, i] = a[i] * SB[j, i] + x[j] * (1.0 - a[i])
            if use_e:
                o_SK[j, i] = a[i] * SK[j, i] + e[j]
            else:
                o_SK[j, i] = a[i] * SK[j, i]
    for i in range(d):
        o_Sba[i] = a[i] * Sba[i] + cc[i]
    # ---- readout: LayerNorm (urutan loop dibuat agar indeks dalam selalu kontigu)
    mh = 0.0
    for i in range(d):
        mh += o_h[i]
    mh /= d
    var = 0.0
    for i in range(d):
        t = o_h[i] - mh
        var += t * t
    inv = 1.0 / np.sqrt(var / d + _EPS_LN)
    for i in range(d):
        hn[i] = (o_h[i] - mh) * inv
    for i in range(d):
        c0[i] = 0.0
    for j in range(d):
        hj = hn[j]
        for i in range(d):
            c0[i] += hj * Wc[j, i]
    if use_mem:
        if query_x:
            # kueri dari token MENTAH (ln(x_t)); h_t adalah campuran berpeluruhan -> satu Wq
            # kecil mungkin tak dapat memulihkan token sekarang. ln(x) ada di slice ekor.
            m0 = 0.0
            for i in range(d):
                m0 += x[i]
            m0 /= d
            v0 = 0.0
            for i in range(d):
                t = x[i] - m0
                v0 += t * t
            iv0 = 1.0 / np.sqrt(v0 / d + _EPS_LN)
            for i in range(d):
                lnx[i] = (x[i] - m0) * iv0
            qsrc = lnx
        else:
            qsrc = hn
        for j in range(dm):
            uq[j] = 0.0
        for i in range(d):
            hi = qsrc[i]
            for j in range(dm):
                uq[j] += hi * Wq[i, j]
        nq = np.sqrt((uq * uq).sum() + _EPS_L2)
        for j in range(dm):
            q[j] = uq[j] / nq
        for i in range(dm):
            s = 0.0
            for j in range(dm):
                s += M[i, j] * q[j]
            am[i] = s
        for i in range(d):
            cf[i] = c0[i]
        for j in range(dm):
            aj = am[j]
            for i in range(d):
                cf[i] += aj * Wm[j, i]
    else:
        for i in range(d):
            cf[i] = c0[i]
    for i in range(ed):
        z1[i] = 0.0
        z2[i] = 0.0
    for j in range(d):
        cj = cf[j]
        for i in range(ed):
            z1[i] += cj * Wg1[j, i]
            z2[i] += cj * Wg2[j, i]
    for i in range(ed):
        sg = 0.5 * (1.0 + np.tanh(0.5 * z1[i]))
        gv[i] = z1[i] * sg * z2[i]
    for i in range(d):
        ff[i] = 0.0
    for j in range(ed):
        gj = gv[j]
        for i in range(d):
            ff[i] += gj * Wo[j, i]
    if use_ad:
        for k in range(r):
            s = 0.0
            for j in range(d):
                s += cf[j] * Af[j, k]
            b1[k] = s
        for i in range(d):
            s = 0.0
            for k in range(r):
                s += b1[k] * Bf[k, i]
            ff[i] += s
    for i in range(d):
        o_y[i] = x[i] + ff[i]
    # ---- tulis memori delta (k, v beku -> M eksogen); hn/uq/q/am/ff bebas dipakai ulang
    if use_mem:
        mx = 0.0
        for i in range(d):
            mx += x[i]
        mx /= d
        vx = 0.0
        for i in range(d):
            t = x[i] - mx
            vx += t * t
        invx = 1.0 / np.sqrt(vx / d + _EPS_LN)
        for i in range(d):
            hn[i] = (x[i] - mx) * invx
            o_kp[i] = hn[i]                     # ln(x_t) disimpan untuk langkah berikutnya
        for j in range(dm):
            s = 0.0
            if hetero:
                for i in range(d):
                    s += kp_prev[i] * Wk[i, j]
            else:
                for i in range(d):
                    s += hn[i] * Wk[i, j]
            uq[j] = s
        nk = np.sqrt((uq * uq).sum() + _EPS_L2)
        for j in range(dm):
            q[j] = uq[j] / nk
        for j in range(dm):
            s = 0.0
            for i in range(d):
                s += hn[i] * Wv[i, j]
            am[j] = s
        beta = beta_max if fix_beta else beta_max * min(1.0, max(0.0, surprise))
        if beta > 0.0:
            for i in range(dm):
                s = 0.0
                for j in range(dm):
                    s += M[i, j] * q[j]
                uq[i] = am[i] - s
            for i in range(dm):
                for j in range(dm):
                    o_M[i, j] = M[i, j] + beta * uq[i] * q[j]
        else:
            for i in range(dm):
                for j in range(dm):
                    o_M[i, j] = M[i, j]
    else:
        for i in range(dm):
            for j in range(dm):
                o_M[i, j] = M[i, j]


@njit(cache=True, fastmath=True)
def flush_rec(h, x, M, gy, Wc, Wq, Wm, Wg1, Wg2, Wo, Af, Bf,
              SWa, SB, SK, Sba, Wa, B, a_in, cc_in, useK, use_mem, use_ad, query_x, gate,
              g, gAf, gBf,
              oWc, oWq, oWm, oWg1, oWg2, oWo, oWa, oB, oba, oK, ox,
              o_gam, o_q, o_gx, scr):
    """Backward readout + kontraksi RTRL; gradien di-scatter langsung ke arena datar `g`.

    gAf/gBf: akumulator gradien parameter cepat (d,r) dan (r,d).
    ox: offset tujuan g_x di `g` (baris E[b] atau flag), atau -1 untuk dibuang.

    query_x=True: kueri dibaca dari ``ln(x_t)`` (token mentah), bukan ``ln(h_t)``. ``x``
    diperlakukan EKSOGEN di jalur ini -- gradien kueri tidak dialirkan ke ``h`` dan ``E``
    tidak menerima gradien lewat jalur kueri (konsisten dengan konvensi stop-grad yang ada).
    """
    d = h.shape[0]
    dm = M.shape[0]
    ed = Wg1.shape[1]
    r = Af.shape[1]
    hn = scr[0:d]
    c0 = scr[d:2 * d]
    cf = scr[2 * d:3 * d]
    ghn = scr[3 * d:4 * d]
    gcf = scr[4 * d:5 * d]
    uq = scr[5 * d:5 * d + dm]
    q = scr[5 * d + dm:5 * d + 2 * dm]
    am = scr[5 * d + 2 * dm:5 * d + 3 * dm]
    tq = scr[5 * d + 3 * dm:5 * d + 4 * dm]
    z1 = scr[5 * d + 4 * dm:5 * d + 4 * dm + ed]
    z2 = scr[5 * d + 4 * dm + ed:5 * d + 4 * dm + 2 * ed]
    gv = scr[5 * d + 4 * dm + 2 * ed:5 * d + 4 * dm + 3 * ed]
    ga = scr[5 * d + 4 * dm + 3 * ed:5 * d + 4 * dm + 4 * ed]
    gb = scr[5 * d + 4 * dm + 4 * ed:5 * d + 4 * dm + 5 * ed]
    gc = scr[5 * d + 4 * dm + 5 * ed:5 * d + 4 * dm + 6 * ed]
    b1 = scr[5 * d + 4 * dm + 6 * ed:5 * d + 4 * dm + 6 * ed + r]
    gb1 = scr[5 * d + 4 * dm + 6 * ed + r:5 * d + 4 * dm + 6 * ed + 2 * r]
    lnx = scr[5 * d + 4 * dm + 6 * ed + 2 * r:6 * d + 4 * dm + 6 * ed + 2 * r]
    # ---------- maju (dihitung ulang)
    mh = 0.0
    for i in range(d):
        mh += h[i]
    mh /= d
    var = 0.0
    for i in range(d):
        t = h[i] - mh
        var += t * t
    inv = 1.0 / np.sqrt(var / d + _EPS_LN)
    for i in range(d):
        hn[i] = (h[i] - mh) * inv
    for i in range(d):
        c0[i] = 0.0
    for j in range(d):
        hj = hn[j]
        for i in range(d):
            c0[i] += hj * Wc[j, i]
    nq = 1.0
    if use_mem:
        if query_x:
            m0 = 0.0
            for i in range(d):
                m0 += x[i]
            m0 /= d
            v0 = 0.0
            for i in range(d):
                t = x[i] - m0
                v0 += t * t
            iv0 = 1.0 / np.sqrt(v0 / d + _EPS_LN)
            for i in range(d):
                lnx[i] = (x[i] - m0) * iv0
            qsrc = lnx
        else:
            qsrc = hn
        for j in range(dm):
            uq[j] = 0.0
        for i in range(d):
            hi = qsrc[i]
            for j in range(dm):
                uq[j] += hi * Wq[i, j]
        nq = np.sqrt((uq * uq).sum() + _EPS_L2)
        for j in range(dm):
            q[j] = uq[j] / nq
            o_q[j] = q[j]                            # kueri baca (untuk backward penulisan)
        for i in range(dm):
            s = 0.0
            for j in range(dm):
                s += M[i, j] * q[j]
            am[i] = s
        for i in range(d):
            cf[i] = c0[i]
        for j in range(dm):
            aj = am[j]
            for i in range(d):
                cf[i] += aj * Wm[j, i]
    else:
        for i in range(d):
            cf[i] = c0[i]
    for i in range(ed):
        z1[i] = 0.0
        z2[i] = 0.0
    for j in range(d):
        cj = cf[j]
        for i in range(ed):
            z1[i] += cj * Wg1[j, i]
            z2[i] += cj * Wg2[j, i]
    for i in range(ed):
        sg = 0.5 * (1.0 + np.tanh(0.5 * z1[i]))
        gv[i] = z1[i] * sg * z2[i]
    if use_ad:
        for k in range(r):
            s = 0.0
            for j in range(d):
                s += cf[j] * Af[j, k]
            b1[k] = s
    # ---------- mundur: GLU
    for i in range(ed):
        s = 0.0
        for j in range(d):
            s += Wo[i, j] * gy[j]
        ga[i] = s                                    # g_gv
    for j in range(ed):
        for i in range(d):
            g[oWo + j * d + i] += gv[j] * gy[i]
    for i in range(d):
        gcf[i] = 0.0
    if use_ad:
        for k in range(r):
            s = 0.0
            for i in range(d):
                s += Bf[k, i] * gy[i]
            gb1[k] = s
        for k in range(r):
            for i in range(d):
                gBf[k, i] += b1[k] * gy[i]
        for j in range(d):
            for k in range(r):
                gAf[j, k] += cf[j] * gb1[k]
        for i in range(d):
            s = 0.0
            for k in range(r):
                s += Af[i, k] * gb1[k]
            gcf[i] += s
    for i in range(ed):
        sg = 0.5 * (1.0 + np.tanh(0.5 * z1[i]))
        gb[i] = ga[i] * z2[i] * (sg + z1[i] * sg * (1.0 - sg))     # g_z1
        gc[i] = ga[i] * z1[i] * sg                                 # g_z2
    for j in range(d):
        for i in range(ed):
            g[oWg1 + j * ed + i] += cf[j] * gb[i]
            g[oWg2 + j * ed + i] += cf[j] * gc[i]
    for j in range(d):
        s = 0.0
        for i in range(ed):
            s += Wg1[j, i] * gb[i] + Wg2[j, i] * gc[i]
        gcf[j] += s
    for i in range(d):
        ghn[i] = 0.0
    # ---------- mundur: jalur memori
    if use_mem:
        for i in range(dm):
            s = 0.0
            for j in range(d):
                s += Wm[i, j] * gcf[j]
            tq[i] = s                                # g_am
            o_gam[i] = s                             # untuk backward penulisan memori
        for i in range(dm):
            for j in range(d):
                g[oWm + i * d + j] += am[i] * gcf[j]
        for j in range(dm):
            s = 0.0
            for i in range(dm):
                s += M[i, j] * tq[i]
            uq[j] = s                                # g_q (uq maju sudah bebas)
        s = 0.0
        for j in range(dm):
            s += uq[j] * q[j]
        for j in range(dm):
            tq[j] = (uq[j] - q[j] * s) / nq          # g_u
        for i in range(d):
            for j in range(dm):
                g[oWq + i * dm + j] += qsrc[i] * tq[j]
        if not query_x:
            # hanya mode h_t: gradien kueri mengalir balik ke h. Pada mode x_t, x eksogen.
            for i in range(d):
                s = 0.0
                for j in range(dm):
                    s += Wq[i, j] * tq[j]
                ghn[i] += s
    # ---------- mundur: Wc + LayerNorm
    for j in range(d):
        for i in range(d):
            g[oWc + j * d + i] += hn[j] * gcf[i]
    for j in range(d):
        s = 0.0
        for i in range(d):
            s += Wc[j, i] * gcf[i]
        ghn[j] += s
    mgh = 0.0
    for i in range(d):
        mgh += ghn[i]
    mgh /= d
    dgh = 0.0
    for i in range(d):
        dgh += ghn[i] * hn[i]
    dgh /= d
    for i in range(d):
        c0[i] = inv * (ghn[i] - mgh - hn[i] * dgh)   # g_h
    # ---- d h_i / d x_j  (masukan menggerakkan gerbang a DAN u, bukan hanya residual)
    #   h_i = a_i*hp_i + (1-a_i)*u_i + e.K  ->  d h_i/d x_j = cc_i*Wa[j,i] + (1-a_i)*B[j,i]
    for j in range(d):
        s = 0.0
        for i in range(d):
            s += c0[i] * (cc_in[i] * Wa[j, i] + (1.0 - a_in[i]) * B[j, i])
        o_gx[j] = s
    # ---------- kontraksi RTRL
    for j in range(d):
        for i in range(d):
            g[oWa + j * d + i] += SWa[j, i] * c0[i]
            g[oB + j * d + i] += SB[j, i] * c0[i]
            if useK:
                g[oK + j * d + i] += SK[j, i] * c0[i]
    for i in range(d):
        g[oba + i] += Sba[i] * c0[i]
    # ---------- g_x = gy + g_h . (dh/dx)     [y = x + f memberi suku gy]
    if ox >= 0:
        if gate:
            for i in range(d):
                g[ox + i] += gy[i] + o_gx[i]
        else:
            for i in range(d):
                g[ox + i] += gy[i]


@njit(cache=True)
def state_cap_scatter(h, SWa, SB, SK, Sba, useK, cap, w, g, oWa, oB, oba, oK):
    """Penalti batas state -> gradien langsung ke parameter rekurensi. Mengembalikan penalti."""
    d = h.shape[0]
    tot = 0.0
    for i in range(d):
        e = abs(h[i]) - cap
        if e > 0.0:
            tot += e * e
            gh = w * 2.0 * e * np.sign(h[i]) / d
            for j in range(d):
                g[oWa + j * d + i] += SWa[j, i] * gh
                g[oB + j * d + i] += SB[j, i] * gh
                if useK:
                    g[oK + j * d + i] += SK[j, i] * gh
            g[oba + i] += Sba[i] * gh
    return w * tot / d


# ------------------------------------------------------------------ AdamW terpadu (slow_update)
@njit(cache=True, fastmath=True)
def adamw_step(gb, adam_m, adam_v, theta, rms_scr,
               lr, b1, b2, bc1_inv, bc2_inv, clip, wd,
               offs, sizes, mat, lrmul, tidx, only_mask):
    """Seluruh inti ``slow_update`` dalam SATU lintasan memori.

    Langkah (URUTAN identik dgn implementasi numpy lama, hanya lintasan digabung):
      1. norm-L2 gb (sqrt((x*x).sum()) -- pembulatan sama dgn numpy di fast_learner)
      2. clip skalar bila gn > clip
      3. only_mask (opsional; dikalikan in-place)
      4. AdamW: adam_m <- b1*adam_m + (1-b1)*gb ; adam_v <- b2*adam_v + (1-b2)*gb^2
      5. theta[i] <- theta[i]*(1-lr*wd) [matriks saja] - lr*lr_mult*upd[i]

    ``bc1_inv``/``bc2_inv`` = 1/(1-b1^t), 1/(1-b2^t) dihitung SEKALI di luar (mahal bila
    per-elemen). ``rms_scr`` buffer panjang ntot (tidak dipakai lagi, disediakan untuk
    perluasan). ``tidx`` daftar indeks nama parameter yang tersentuh; ``offs/sizes/mat/
    lrmul`` array paralel per indeks nama (lihat FastLearner._build_slow_tables).

    Catatan fastmath: ``sqrt`` dan pembagian boleh dipetakan ke rsqrt/approx oleh LLVM.
    Penyimpangan vs numpy lama ~1e-15 relatif (uji: scripts/proto_slow_kernel*.py).
    """
    n = gb.shape[0]
    s = 0.0
    for i in range(n):
        gi = gb[i]
        s += gi * gi
    gn = np.sqrt(s)
    if gn > clip:
        sc = clip / gn
        for i in range(n):
            gb[i] *= sc
    if only_mask is not None:
        for i in range(n):
            gb[i] *= only_mask[i]
    c1 = 1.0 - b1
    c2 = 1.0 - b2
    for i in range(n):
        gi = gb[i]
        adam_m[i] = adam_m[i] * b1 + gi * c1
        adam_v[i] = adam_v[i] * b2 + gi * gi * c2
    dcy = 1.0 - lr * wd
    ne = tidx.shape[0]
    for k in range(ne):
        nm = tidx[k]
        o = offs[nm]
        sz = sizes[nm]
        ml = lr * lrmul[nm]
        d = mat[nm] != 0
        for j in range(sz):
            idx = o + j
            upd = (adam_m[idx] * bc1_inv) / (np.sqrt(adam_v[idx] * bc2_inv) + 1e-8)
            v = theta[idx]
            if d:
                v *= dcy
            theta[idx] = v - ml * upd


# =====================================================================================
# Kernel TERPADU LINTAS-LAPISAN (fused) -- target ~38k B/s.
#
# Motivasi: dispatch njit tumbuh dengan ARITY (no-op 1-arg 0,23 us vs 30-arg 2,53 us).
# step_fwd punya 42 arg -> ~4 us hanya untuk menyusun tuple + dispatch, dipanggil n_layers
# kali per byte, lalu kepala dipanggil terpisah. Kernel di bawah:
#   1. menerima SATU `bundle` (tuple parameter tetap) -> arity panggilan turun ~6x;
#   2. menyimpan SELURUH state lapisan di array datar berstride tetap (satu alokasi);
#   3. menjalankan loop lapisan di DALAM numba (y_l menjadi x_{l+1});
#   4. diakhiri kepala linear + gradien softmax-CE dalam panggilan yang SAMA.
#
# Kontrak: urutan operasi float disamakan PERSIS dengan step_fwd/head_fwd_lin supaya hasil
# bit-identik (penting: chaos online -> beda 1e-15 bisa menggeser BPB 0,06 bit).
# =====================================================================================

def state_size_per_layer(d, dm):
    """Jumlah float state SATU lapisan di arena datar.

    Layout (urutan tetap, lihat FUSED_KERNEL.md):
        h(d) S_Wa(d*d) S_B(d*d) S_K(d*d) S_ba(d) y(d)
        M(dm*dm) [memori SEBELUM tulis -- dibaca saat backward]
        Mpost(dm*dm) [memori SESUDAH tulis -- state yang mengalir ke langkah berikut]
        kp(d) a(d) cc(d)
    """
    return 6 * d + 3 * d * d + 2 * dm * dm


def state_offsets(n_layers, d, dm):
    """Offset tiap field state per lapisan (referensi; kernel memakai aritmetika langsung)."""
    s = state_size_per_layer(d, dm)
    off = {
        "h": 0, "SWa": d, "SB": d + d * d, "SK": d + 2 * d * d, "Sba": d + 3 * d * d,
        "y": d + 3 * d * d + d, "M": d + 3 * d * d + 2 * d,
        "Mpost": d + 3 * d * d + 2 * d + dm * dm,
        "kp": d + 3 * d * d + 2 * d + 2 * dm * dm,
        "a": d + 3 * d * d + 3 * d + 2 * dm * dm,
        "cc": d + 3 * d * d + 4 * d + 2 * dm * dm,
    }
    return s, off


def fused_act_size(d, dm, ed, r):
    """Ukuran buffer aktivasi per lapisan untuk fused_bwd (menghindari hitung ulang pass maju).

    Layout per lapisan: [hn(d), c0(d), cf(d), z1(ed), z2(ed), gv(ed), q(dm), am(dm), b1(r),
    lnx(d), inv(1)]
    `lnx` = ln(x_t) dipakai sebagai qsrc bila mem_query_x; `inv` = 1/sqrt(var/d+eps) LayerNorm.
    """
    return 7 * d + 3 * ed + 2 * dm + r + 1


def fused_act_offsets(nl, d, dm, ed, r):
    """(per_layer_size, offsets_dict) untuk buffer aktivasi fused."""
    s = fused_act_size(d, dm, ed, r)
    o = {}
    p = 0
    for nm_, sz in (("hn", d), ("c0", d), ("cf", d), ("z1", ed), ("z2", ed),
                    ("gv", ed), ("q", dm), ("am", dm), ("b1", r), ("lnx", d), ("inv", 1)):
        o[nm_] = p
        p += sz
    return s, o


@njit(cache=True, fastmath=True)
def fused_fwd(ARGS):
    """Maju SELURUH tumpukan + kepala linear + CE dalam SATU panggilan njit.

    ARGS = (bundle, st, loff, xin, out, scr, act, aoff, logits_prev, prev_byte) satu tuple
    (biaya dispatch numba tumbuh dgn ARITY; tuple tetap murah).
    `out` : [y_top(d), logits(V), a0(d), cc0(d), (a_l,cc_l) per l, CE(1)]
    CE prekuensial byte ini = -log softmax(logits_prev)[ce_target], ditulis ke `out[-1]`.
    ce_target = byte INI (prediksi dari logits byte SEBELUMNYA). -1.0 bila ce_target < 0.
    """
    bundle = ARGS[0]
    st = ARGS[1]
    loff = ARGS[2]
    xin = ARGS[3]
    out = ARGS[4]
    scr = ARGS[5]
    act = ARGS[6]
    aoff = ARGS[7]
    logits_prev = ARGS[8]
    ce_target = ARGS[9]          # byte INI (target CE prekuensial thd logits sebelumnya)
    d = bundle[0]
    dm = bundle[1]
    ed = bundle[2]
    r = bundle[3]
    V = bundle[4]
    nl = bundle[5]
    use_mem = bundle[6]
    use_ad = bundle[7]
    query_x = bundle[8]
    hetero = bundle[9]
    fix_beta = bundle[10]
    use_e_g = bundle[11]
    beta_max = bundle[12]
    # par[l] = (Wa, ba, B, K, Wc, Wq, Wm, Wg1, Wg2, Wo, Af, Bf, Wk, Wv)
    par = bundle[13]
    # slice scratch (dipakai ulang per lapisan)
    a = scr[0:d]
    u = scr[d:2 * d]
    cc = scr[2 * d:3 * d]
    hn = scr[3 * d:4 * d]
    c0 = scr[4 * d:5 * d]
    cf = scr[5 * d:6 * d]
    ff = scr[6 * d:7 * d]
    uq = scr[7 * d:7 * d + dm]
    q = scr[7 * d + dm:7 * d + 2 * dm]
    am = scr[7 * d + 2 * dm:7 * d + 3 * dm]
    z1 = scr[7 * d + 3 * dm:7 * d + 3 * dm + ed]
    z2 = scr[7 * d + 3 * dm + ed:7 * d + 3 * dm + 2 * ed]
    gv = scr[7 * d + 3 * dm + 2 * ed:7 * d + 3 * dm + 3 * ed]
    b1 = scr[7 * d + 3 * dm + 3 * ed:7 * d + 3 * dm + 3 * ed + r]
    lnx = scr[7 * d + 3 * dm + 3 * ed + r:8 * d + 3 * dm + 3 * ed + r]
    lgx = lnx                                     # dipakai ulang di fase tulis memori
    ssz = 6 * d + 3 * d * d + 2 * dm * dm
    # ---- buffer aktivasi (per lapisan) untuk fused_bwd: hindari hitung ulang pass maju.
    # aoff = ukuran per lapisan; irisan per-lapisan dibuat di dalam loop.
    # ---- loop lapisan
    for l in range(nl):
        abase = l * aoff
        Ahn = act[abase:abase + d]
        Ac0 = act[abase + d:abase + 2 * d]
        Acf = act[abase + 2 * d:abase + 3 * d]
        Az1 = act[abase + 3 * d:abase + 3 * d + ed]
        Az2 = act[abase + 3 * d + ed:abase + 3 * d + 2 * ed]
        Agv = act[abase + 3 * d + 2 * ed:abase + 3 * d + 3 * ed]
        Ab1 = act[abase + 3 * d + 3 * ed:abase + 3 * d + 3 * ed + r]
        Aq = act[abase + 3 * d + 3 * ed + r:abase + 3 * d + 3 * ed + r + dm]
        Aam = act[abase + 3 * d + 3 * ed + r + dm:abase + 3 * d + 3 * ed + r + 2 * dm]
        Alnx = act[abase + 3 * d + 3 * ed + r + 2 * dm:abase + 3 * d + 3 * ed + r + 2 * dm + d]
        Wa, ba, B, K, Wc, Wq, Wm, Wg1, Wg2, Wo, Af, Bf, Wk, Wv = par[l]
        base = l * ssz
        h = st[base:base + d]
        SWa = st[base + d:base + d + d * d].reshape(d, d)
        SB = st[base + d + d * d:base + d + 2 * d * d].reshape(d, d)
        SK = st[base + d + 2 * d * d:base + d + 3 * d * d].reshape(d, d)
        Sba = st[base + d + 3 * d * d:base + d + 3 * d * d + d]
        y = st[base + d + 3 * d * d + d:base + d + 3 * d * d + 2 * d]
        # M  = memori SEBELUM tulis (dibaca utk readout, dan disimpan utk backward).
        # Mpost = memori SESUDAH tulis (state yg mengalir ke langkah berikut).
        Mpre = st[base + d + 3 * d * d + 2 * d:base + d + 3 * d * d + 2 * d + dm * dm].reshape(dm, dm)
        M = Mpre                                   # alias baca (delta-rule memakai M sebelum)
        Mpost = st[base + d + 3 * d * d + 2 * d + dm * dm:
                   base + d + 3 * d * d + 2 * d + 2 * dm * dm].reshape(dm, dm)
        kp = st[base + d + 3 * d * d + 2 * d + 2 * dm * dm:base + d + 3 * d * d + 3 * d + 2 * dm * dm]
        aa = st[base + d + 3 * d * d + 3 * d + 2 * dm * dm:base + d + 3 * d * d + 4 * d + 2 * dm * dm]
        ccs = st[base + d + 3 * d * d + 4 * d + 2 * dm * dm:base + d + 3 * d * d + 5 * d + 2 * dm * dm]
        x = xin if l == 0 else st[(l - 1) * ssz + d + 3 * d * d + d:
                                  (l - 1) * ssz + d + 3 * d * d + 2 * d]   # y lapisan bawah
        # ---- rekurensi + jejak (urutan PERSIS step_fwd)
        for i in range(d):
            s = ba[i]
            for j in range(d):
                s += x[j] * Wa[j, i]
            ai = 0.5 * (1.0 + np.tanh(0.5 * s))
            a[i] = ai
            t = 0.0
            for j in range(d):
                t += x[j] * B[j, i]
            u[i] = t
            cc[i] = (h[i] - t) * ai * (1.0 - ai)
        for i in range(d):
            h[i] = a[i] * h[i] + (1.0 - a[i]) * u[i]
            aa[i] = a[i]
            ccs[i] = cc[i]
        for j in range(d):
            for i in range(d):
                SWa[j, i] = a[i] * SWa[j, i] + x[j] * cc[i]
                SB[j, i] = a[i] * SB[j, i] + x[j] * (1.0 - a[i])
                SK[j, i] = a[i] * SK[j, i]
        for i in range(d):
            Sba[i] = a[i] * Sba[i] + cc[i]
        # ---- readout: LN
        mh = 0.0
        for i in range(d):
            mh += h[i]
        mh /= d
        var = 0.0
        for i in range(d):
            t = h[i] - mh
            var += t * t
        inv = 1.0 / np.sqrt(var / d + _EPS_LN)
        act[abase + aoff - 1] = inv
        for i in range(d):
            hn[i] = (h[i] - mh) * inv
        for i in range(d):
            Ahn[i] = hn[i]
        for i in range(d):
            c0[i] = 0.0
        for j in range(d):
            hj = hn[j]
            for i in range(d):
                c0[i] += hj * Wc[j, i]
        for i in range(d):
            Ac0[i] = c0[i]
        if use_mem:
            if query_x:
                m0 = 0.0
                for i in range(d):
                    m0 += x[i]
                m0 /= d
                v0 = 0.0
                for i in range(d):
                    t = x[i] - m0
                    v0 += t * t
                iv0 = 1.0 / np.sqrt(v0 / d + _EPS_LN)
                for i in range(d):
                    lnx[i] = (x[i] - m0) * iv0
                for i in range(d):
                    Alnx[i] = lnx[i]
                qsrc = lnx
            else:
                qsrc = hn
            for j in range(dm):
                uq[j] = 0.0
            for i in range(d):
                hi = qsrc[i]
                for j in range(dm):
                    uq[j] += hi * Wq[i, j]
            nq = np.sqrt((uq * uq).sum() + _EPS_L2)
            for j in range(dm):
                q[j] = uq[j] / nq
            for i in range(dm):
                s = 0.0
                for j in range(dm):
                    s += M[i, j] * q[j]
                am[i] = s
            for i in range(dm):
                Aq[i] = q[i]
                Aam[i] = am[i]
            for i in range(d):
                cf[i] = c0[i]
            for j in range(dm):
                aj = am[j]
                for i in range(d):
                    cf[i] += aj * Wm[j, i]
        else:
            for i in range(d):
                cf[i] = c0[i]
        for i in range(d):
            Acf[i] = cf[i]
        # ---- GLU
        for i in range(ed):
            z1[i] = 0.0
            z2[i] = 0.0
        for j in range(d):
            cj = cf[j]
            for i in range(ed):
                z1[i] += cj * Wg1[j, i]
                z2[i] += cj * Wg2[j, i]
        for i in range(ed):
            Az1[i] = z1[i]
            Az2[i] = z2[i]
            sg = 0.5 * (1.0 + np.tanh(0.5 * z1[i]))
            gv[i] = z1[i] * sg * z2[i]
        for i in range(ed):
            Agv[i] = gv[i]
        for i in range(d):
            ff[i] = 0.0
        for j in range(ed):
            gj = gv[j]
            for i in range(d):
                ff[i] += gj * Wo[j, i]
        if use_ad:
            for k in range(r):
                s = 0.0
                for j in range(d):
                    s += cf[j] * Af[j, k]
                b1[k] = s
            for k in range(r):
                Ab1[k] = b1[k]
            for i in range(d):
                s = 0.0
                for k in range(r):
                    s += b1[k] * Bf[k, i]
                ff[i] += s
        for i in range(d):
            y[i] = x[i] + ff[i]
        # ---- tulis memori delta (k, v beku -> M eksogen); ln(x_t) disimpan untuk langkah brik
        if use_mem:
            mx = 0.0
            for i in range(d):
                mx += x[i]
            mx /= d
            vx = 0.0
            for i in range(d):
                t = x[i] - mx
                vx += t * t
            invx = 1.0 / np.sqrt(vx / d + _EPS_LN)
            # kp lama harus dibaca SEBELUM ditimpa (hetero memakai kp_{t-1} sebagai kunci);
            # ln(x_t) masuk ke hn scratch, kp ditulis SETELAH pembacaan kunci.
            for i in range(d):
                lgx[i] = (x[i] - mx) * invx        # ln(x_t) sementara (scratch)
            for j in range(dm):
                s = 0.0
                if hetero:
                    for i in range(d):
                        s += kp[i] * Wk[i, j]
                else:
                    for i in range(d):
                        s += lgx[i] * Wk[i, j]
                uq[j] = s
            nk = np.sqrt((uq * uq).sum() + _EPS_L2)
            for j in range(dm):
                q[j] = uq[j] / nk
            for j in range(dm):
                s = 0.0
                for i in range(d):
                    s += lgx[i] * Wv[i, j]
                am[j] = s
            for i in range(d):
                kp[i] = lgx[i]                     # ln(x_t) disimpan untuk langkah berikutnya
            beta = beta_max
            if beta > 0.0:
                for i in range(dm):
                    s = 0.0
                    for j in range(dm):
                        s += M[i, j] * q[j]        # baca M SEBELUM tulis
                    uq[i] = am[i] - s
                for i in range(dm):
                    for j in range(dm):
                        Mpost[i, j] = M[i, j] + beta * uq[i] * q[j]
            else:
                for i in range(dm):
                    for j in range(dm):
                        Mpost[i, j] = M[i, j]
        else:
            for i in range(dm):
                for j in range(dm):
                    Mpost[i, j] = 0.0
    # ---- kepala linear pada y lapisan atas; masukan = y_top @ P[(top,1)]
    ytop = st[(nl - 1) * ssz + d + 3 * d * d + d:(nl - 1) * ssz + d + 3 * d * d + 2 * d]
    W_out, byte_bias, Ptop = bundle[14], bundle[15], bundle[16]
    p1 = scr[8 * d + 3 * dm + 3 * ed + r:8 * d + 3 * dm + 3 * ed + r + d]
    for i in range(d):
        s = 0.0
        for j in range(d):
            s += ytop[j] * Ptop[j, i]
        p1[i] = s
    nrm = 0.0
    for i in range(d):
        nrm += p1[i] * p1[i]
    nrm = np.sqrt(nrm + _EPS_L2)
    pv = scr[8 * d + 3 * dm + 3 * ed + r + d:8 * d + 3 * dm + 3 * ed + r + 2 * d]
    for i in range(d):
        pv[i] = p1[i] / nrm
    lg = out[d:d + V]
    for i in range(V):
        s = 0.0
        for j in range(d):
            s += W_out[i, j] * pv[j]
        lg[i] = s + byte_bias[i]
    # ---- CE prekuensial byte ini (dari logits byte SEBELUMNYA, target prev_byte).
    # Difusi ce_val ke dalam kernel -> hemat 1 dispatch ~9 us/byte. out[-1] = CE (-1 bila tak ada).
    if ce_target >= 0:
        mx = -1e300
        for i in range(V):
            if logits_prev[i] > mx:
                mx = logits_prev[i]
        sm = 0.0
        for i in range(V):
            sm += np.exp(logits_prev[i] - mx)
        out[out.shape[0] - 1] = float(mx + np.log(sm) - logits_prev[ce_target])
    else:
        out[out.shape[0] - 1] = -1.0


# =====================================================================================
# Fused backward: kepala + n_layers x flush_rec dalam SATU panggilan.
#
# Kontrak: untuk loss byte murni (only_byte_loss + deep_grad), hanya lapisan atas
# menerima gradien dari kepala. g_x lapisan l (dari flush_rec) dialirkan sebagai gy
# lapisan l-1 (sama seperti _flush deep=True: b.gh += gy + o_gx).
# =====================================================================================

@njit(cache=True, fastmath=True)
def fused_bwd(ARGS):
    """Mundur tumpukan untuk loss byte (di dalam: kepala + n_layers x flush_rec).

    ARGS = (bundle, st, loff, xin, glo, byte, ox0, G, gE, scr, act, aoff, recompute) satu tuple:
    biaya dispatch numba tumbuh dgn ARITY, tuple tetap murah -> hemat ~3 us/byte.
    `act`/`aoff`: buffer aktivasi dari fused_fwd -> backward TIDAK hitung ulang pass maju.
    `recompute`: 1 -> abaikan `act`, hitung ulang pass maju (WAJIB pada backward pertama
    setelah slow_update; lihat komentar di badan fungsi).
    """
    bundle = ARGS[0]
    st = ARGS[1]
    loff = ARGS[2]
    xin = ARGS[3]
    logits_prev = ARGS[4]        # logits byte sebelumnya (dari fwd byte lalu)
    byte = ARGS[5]
    ox0 = ARGS[6]
    G = ARGS[7]
    gE = ARGS[8]
    scr = ARGS[9]
    act = ARGS[10]
    aoff = ARGS[11]
    recompute = ARGS[12]   # 1 -> hitung ulang pass maju (WAJIB pada backward pertama pasca-SU)
    # bundle[17]=useK, bundle[18]=deep_gate (flag). G = arena datar gradien (ntot,).
    # gE = baris kamus E (vocab*d) atau None diganti flag via o_flag di luar.
    d = bundle[0]
    dm = bundle[1]
    ed = bundle[2]
    r = bundle[3]
    V = bundle[4]
    nl = bundle[5]
    use_mem = bundle[6]
    use_ad = bundle[7]
    query_x = bundle[8]
    useK = bundle[17]
    deep_gate = bundle[18]
    par = bundle[13]
    W_out = bundle[14]
    byte_bias = bundle[15]
    Ptop = bundle[16]
    ssz = 6 * d + 3 * d * d + 2 * dm * dm
    # scratch backward (disusun ulang agar tidak tumpang tindih)
    hn = scr[0:d]
    c0 = scr[d:2 * d]
    cf = scr[2 * d:3 * d]
    ghn = scr[3 * d:4 * d]
    gcf = scr[4 * d:5 * d]
    uq = scr[5 * d:5 * d + dm]
    q = scr[5 * d + dm:5 * d + 2 * dm]
    am = scr[5 * d + 2 * dm:5 * d + 3 * dm]
    tq = scr[5 * d + 3 * dm:5 * d + 4 * dm]
    z1 = scr[5 * d + 4 * dm:5 * d + 4 * dm + ed]
    z2 = scr[5 * d + 4 * dm + ed:5 * d + 4 * dm + 2 * ed]
    gv = scr[5 * d + 4 * dm + 2 * ed:5 * d + 4 * dm + 3 * ed]
    ga = scr[5 * d + 4 * dm + 3 * ed:5 * d + 4 * dm + 4 * ed]
    gb = scr[5 * d + 4 * dm + 4 * ed:5 * d + 4 * dm + 5 * ed]
    gc = scr[5 * d + 4 * dm + 5 * ed:5 * d + 4 * dm + 6 * ed]
    b1 = scr[5 * d + 4 * dm + 6 * ed:5 * d + 4 * dm + 6 * ed + r]
    gb1 = scr[5 * d + 4 * dm + 6 * ed + r:5 * d + 4 * dm + 6 * ed + 2 * r]
    lnx = scr[5 * d + 4 * dm + 6 * ed + 2 * r:6 * d + 4 * dm + 6 * ed + 2 * r]
    gy = scr[6 * d + 4 * dm + 6 * ed + 2 * r:7 * d + 4 * dm + 6 * ed + 2 * r]
    gxp = scr[7 * d + 4 * dm + 6 * ed + 2 * r:8 * d + 4 * dm + 6 * ed + 2 * r]
    pv = scr[8 * d + 4 * dm + 6 * ed + 2 * r:9 * d + 4 * dm + 6 * ed + 2 * r]
    goft = 9 * d + 4 * dm + 6 * ed + 2 * r
    gs = scr[goft:goft + V]
    glo = scr[goft + V + 3 * d:goft + 2 * V + 3 * d]
    gp = scr[goft + V:goft + V + d]
    o_gam_top = scr[goft + V + d:goft + V + 2 * d]
    fp_top = scr[goft + V + 2 * d:goft + V + 3 * d]
    # ---------- gradien CE thd logits byte sebelumnya (dulu softmax_grad_target terpisah;
    # difusi ke dalam kernel -> hemat 1 dispatch ~13 us/byte). w = 1.0 (only_byte_loss).
    mx = -1e300
    for i in range(V):
        if logits_prev[i] > mx:
            mx = logits_prev[i]
    sm = 0.0
    for i in range(V):
        e = np.exp(logits_prev[i] - mx)
        glo[i] = e
        sm += e
    inv_sm = 1.0 / sm
    for i in range(V):
        glo[i] = glo[i] * inv_sm
    glo[byte] -= 1.0
    # ---------- kepala linear mundur (head_bwd_lin_acc, tanpa alokasi)
    ytop = st[(nl - 1) * ssz + d + 3 * d * d + d:(nl - 1) * ssz + d + 3 * d * d + 2 * d]
    for i in range(d):
        s = 0.0
        for j in range(d):
            s += ytop[j] * Ptop[j, i]
        fp_top[i] = s
    nrm = 0.0
    for i in range(d):
        nrm += fp_top[i] * fp_top[i]
    nrm = np.sqrt(nrm + _EPS_L2)
    for i in range(d):
        pv[i] = fp_top[i] / nrm
    if bundle[19] >= 0:                                  # oWout: scatter W_out
        oW = bundle[19]
        for i in range(V):
            gi = glo[i]
            base = oW + i * d
            for j in range(d):
                G[base + j] += gi * pv[j]
    oB = bundle[20]                                      # offset byte_bias
    for i in range(V):
        G[oB + i] += glo[i]
    # g_p = (g_s - pv*(g_s.pv))/nrm ; g_s = W_out^T @ glo
    for j in range(d):
        gs[j] = 0.0
    for i in range(V):
        gi = glo[i]
        for j in range(d):
            gs[j] += gi * W_out[i, j]
    dg = 0.0
    for j in range(d):
        dg += gs[j] * pv[j]
    for j in range(d):
        gp[j] = (gs[j] - pv[j] * dg) / nrm
    # gy_top = P @ gp   (head_bwd_lin_acc: g_y = _mv(P, g_p))
    for i in range(d):
        s = 0.0
        for j in range(d):
            s += Ptop[i, j] * gp[j]
        gy[i] = s
    # gP_top = outer(ytop, gp) -> offset bundle[22]
    oP = bundle[21]
    if oP >= 0:
        for i in range(d):
            yi = ytop[i]
            for j in range(d):
                G[oP + i * d + j] += yi * gp[j]
    # ---------- loop lapisan (puncak -> dasar)
    for l in range(nl - 1, -1, -1):
        Wa, ba, B, K, Wc, Wq, Wm, Wg1, Wg2, Wo, Af, Bf, Wk, Wv = par[l]
        base = l * ssz
        h = st[base:base + d]
        SWa = st[base + d:base + d + d * d].reshape(d, d)
        SB = st[base + d + d * d:base + d + 2 * d * d].reshape(d, d)
        SK = st[base + d + 2 * d * d:base + d + 3 * d * d].reshape(d, d)
        Sba = st[base + d + 3 * d * d:base + d + 3 * d * d + d]
        y = st[base + d + 3 * d * d + d:base + d + 3 * d * d + 2 * d]
        M = st[base + d + 3 * d * d + 2 * d:base + d + 3 * d * d + 2 * d + dm * dm].reshape(dm, dm)
        aa = st[base + d + 3 * d * d + 3 * d + 2 * dm * dm:base + d + 3 * d * d + 4 * d + 2 * dm * dm]
        ccs = st[base + d + 3 * d * d + 4 * d + 2 * dm * dm:base + d + 3 * d * d + 5 * d + 2 * dm * dm]
        x = xin if l == 0 else st[(l - 1) * ssz + d + 3 * d * d + d:
                                  (l - 1) * ssz + d + 3 * d * d + 2 * d]
        lo = loff[l]
        oWa = lo[0]; oba = lo[1]; oB = lo[2]; oK = lo[3]
        oWc = lo[4]; oWq = lo[5]; oWm = lo[6]; oWg1 = lo[7]; oWg2 = lo[8]; oWo = lo[9]
        # ---------- baca aktivasi yang disimpan fused_fwd (TIDAK hitung ulang pass maju)
        abase = l * aoff
        Ahn = act[abase:abase + d]
        Ac0 = act[abase + d:abase + 2 * d]
        Acf = act[abase + 2 * d:abase + 3 * d]
        Az1 = act[abase + 3 * d:abase + 3 * d + ed]
        Az2 = act[abase + 3 * d + ed:abase + 3 * d + 2 * ed]
        Agv = act[abase + 3 * d + 2 * ed:abase + 3 * d + 3 * ed]
        Ab1 = act[abase + 3 * d + 3 * ed:abase + 3 * d + 3 * ed + r]
        Aq = act[abase + 3 * d + 3 * ed + r:abase + 3 * d + 3 * ed + r + dm]
        Aam = act[abase + 3 * d + 3 * ed + r + dm:abase + 3 * d + 3 * ed + r + 2 * dm]
        Alnx = act[abase + 3 * d + 3 * ed + r + 2 * dm:abase + 3 * d + 3 * ed + r + 2 * dm + d]
        inv = act[abase + aoff - 1]
        for i in range(dm):
            q[i] = Aq[i]
            am[i] = Aam[i]
        # ---------- pilih sumber aktivasi maju
        #   recompute == 0 -> baca aktivasi TERSIMPAN (fused_fwd). CEPAT, tetapi HANYA sah bila
        #     parameter tidak berubah antara langkah maju dan langkah mundur ini.
        #   recompute != 0 -> HITUNG ULANG pass maju (identik flush_rec). WAJIB dipakai untuk
        #     langkah mundur PERTAMA setelah `slow_update` (theta berubah, aktivasi tersimpan
        #     menjadi basi: terukur galat gradien 1,6e-4..1,2e-3). Biaya: satu pass maju per SU.
        if recompute != 0:
            mh = 0.0
            for i in range(d):
                mh += h[i]
            mh /= d
            varr = 0.0
            for i in range(d):
                t = h[i] - mh
                varr += t * t
            inv = 1.0 / np.sqrt(varr / d + _EPS_LN)
            for i in range(d):
                hn[i] = (h[i] - mh) * inv
            for i in range(d):
                c0[i] = 0.0
            for j in range(d):
                hj = hn[j]
                for i in range(d):
                    c0[i] += hj * Wc[j, i]
            if use_mem:
                if query_x:
                    m0 = 0.0
                    for i in range(d):
                        m0 += x[i]
                    m0 /= d
                    v0 = 0.0
                    for i in range(d):
                        t = x[i] - m0
                        v0 += t * t
                    iv0 = 1.0 / np.sqrt(v0 / d + _EPS_LN)
                    for i in range(d):
                        lnx[i] = (x[i] - m0) * iv0
                    qsrc = lnx
                else:
                    qsrc = hn
                for j in range(dm):
                    uq[j] = 0.0
                for i in range(d):
                    hi = qsrc[i]
                    for j in range(dm):
                        uq[j] += hi * Wq[i, j]
                qn = np.sqrt((uq * uq).sum() + _EPS_L2)
                for j in range(dm):
                    q[j] = uq[j] / qn
                for i in range(dm):
                    s = 0.0
                    for j in range(dm):
                        s += M[i, j] * q[j]
                    am[i] = s
                for i in range(d):
                    cf[i] = c0[i]
                for j in range(dm):
                    aj = am[j]
                    for i in range(d):
                        cf[i] += aj * Wm[j, i]
            else:
                for i in range(d):
                    cf[i] = c0[i]
            for i in range(ed):
                z1[i] = 0.0
                z2[i] = 0.0
            for j in range(d):
                cj = cf[j]
                for i in range(ed):
                    z1[i] += cj * Wg1[j, i]
                    z2[i] += cj * Wg2[j, i]
            for i in range(ed):
                sg = 0.5 * (1.0 + np.tanh(0.5 * z1[i]))
                gv[i] = z1[i] * sg * z2[i]
        else:
            for i in range(d):
                hn[i] = Ahn[i]
                c0[i] = Ac0[i]
                cf[i] = Acf[i]
            for i in range(ed):
                z1[i] = Az1[i]
                z2[i] = Az2[i]
                gv[i] = Agv[i]
        for j in range(dm):
            uq[j] = 0.0
        nq = 1.0
        if use_mem:
            # nq = |Wq^T qsrc + eps| -- perlukan utk backward normalisasi kueri
            if query_x:
                # ln(x_t) hanya bergantung pada x (eksogen) -> nilai tersimpan selalu valid.
                qsrc = Alnx
                for i in range(d):
                    lnx[i] = Alnx[i]
            else:
                qsrc = hn
            for i in range(d):
                hi = qsrc[i]
                for j in range(dm):
                    uq[j] += hi * Wq[i, j]
            nq = np.sqrt((uq * uq).sum() + _EPS_L2)
        if use_ad:
            for k in range(r):
                b1[k] = Ab1[k]
        # ---------- mundur: GLU
        for i in range(ed):
            s = 0.0
            for j in range(d):
                s += Wo[i, j] * gy[j]
            ga[i] = s
        for j in range(ed):
            for i in range(d):
                G[oWo + j * d + i] += gv[j] * gy[i]
        for i in range(d):
            gcf[i] = 0.0
        if use_ad:
            for k in range(r):
                s = 0.0
                for i in range(d):
                    s += Bf[k, i] * gy[i]
                gb1[k] = s
            for i in range(d):
                s = 0.0
                for k in range(r):
                    s += Af[i, k] * gb1[k]
                gcf[i] += s
        for i in range(ed):
            sg = 0.5 * (1.0 + np.tanh(0.5 * z1[i]))
            gb[i] = ga[i] * z2[i] * (sg + z1[i] * sg * (1.0 - sg))
            gc[i] = ga[i] * z1[i] * sg
        for j in range(d):
            for i in range(ed):
                G[oWg1 + j * ed + i] += cf[j] * gb[i]
                G[oWg2 + j * ed + i] += cf[j] * gc[i]
        for j in range(d):
            s = 0.0
            for i in range(ed):
                s += Wg1[j, i] * gb[i] + Wg2[j, i] * gc[i]
            gcf[j] += s
        for i in range(d):
            ghn[i] = 0.0
        # ---------- mundur: jalur memori
        if use_mem:
            for i in range(dm):
                s = 0.0
                for j in range(d):
                    s += Wm[i, j] * gcf[j]
                tq[i] = s
                if l == nl - 1:
                    o_gam_top[i] = s
            for i in range(dm):
                for j in range(d):
                    G[oWm + i * d + j] += am[i] * gcf[j]
            for j in range(dm):
                s = 0.0
                for i in range(dm):
                    s += M[i, j] * tq[i]
                uq[j] = s
            s = 0.0
            for j in range(dm):
                s += uq[j] * q[j]
            for j in range(dm):
                tq[j] = (uq[j] - q[j] * s) / nq
            for i in range(d):
                for j in range(dm):
                    G[oWq + i * dm + j] += qsrc[i] * tq[j]
            if not query_x:
                for i in range(d):
                    s = 0.0
                    for j in range(dm):
                        s += Wq[i, j] * tq[j]
                    ghn[i] += s
        # ---------- mundur: Wc + LayerNorm
        for j in range(d):
            for i in range(d):
                G[oWc + j * d + i] += hn[j] * gcf[i]
        for j in range(d):
            s = 0.0
            for i in range(d):
                s += Wc[j, i] * gcf[i]
            ghn[j] += s
        mgh = 0.0
        for i in range(d):
            mgh += ghn[i]
        mgh /= d
        dgh = 0.0
        for i in range(d):
            dgh += ghn[i] * hn[i]
        dgh /= d
        for i in range(d):
            c0[i] = inv * (ghn[i] - mgh - hn[i] * dgh)
        # ---------- d h_i / d x_j
        for j in range(d):
            s = 0.0
            for i in range(d):
                s += c0[i] * (ccs[i] * Wa[j, i] + (1.0 - aa[i]) * B[j, i])
            gxp[j] = s
        # ---------- kontraksi RTRL
        for j in range(d):
            for i in range(d):
                G[oWa + j * d + i] += SWa[j, i] * c0[i]
                G[oB + j * d + i] += SB[j, i] * c0[i]
                if useK:
                    G[oK + j * d + i] += SK[j, i] * c0[i]
        for i in range(d):
            G[oba + i] += Sba[i] * c0[i]
        # ---------- g_x -> gy lapisan bawah (deep)
        if l > 0:
            for i in range(d):
                gy[i] = (gxp[i] + gy[i]) if deep_gate else gy[i]
        else:
            # lapisan 0: g_x = gy + o_gx -> baris E[b] (ox0 >= 0) atau dibuang
            if ox0 >= 0:
                for i in range(d):
                    G[ox0 + i] += (gxp[i] + gy[i]) if deep_gate else gy[i]


@njit(cache=True, fastmath=True)
def fused_promote_mem(st, d, dm, nl):
    """Salin Mpost -> Mpre tiap lapisan (state memori mengalir ke langkah berikutnya)."""
    ssz = 6 * d + 3 * d * d + 2 * dm * dm
    moff = d + 3 * d * d + 2 * d
    for l in range(nl):
        base = l * ssz
        for j in range(dm * dm):
            st[base + moff + j] = st[base + moff + dm * dm + j]


@njit(cache=True, fastmath=True)
def fused_decay_traces(st, d, dm, nl, factor):
    """Skala jejak RTRL (SWa, SB, SK, Sba) dengan `factor` -- padanan LayerState.decay_traces.

    WAJIB dipanggil `slow_update` (sama seperti jalur lama) agar kontraksi RTRL pada byte
    berikutnya memakai jejak yang konsisten. Tanpa ini, gradien arena menyimpang setelah
    slow_update pertama (terukur 0,065 pada SU1, s_decay=0.5).
    """
    ssz = 6 * d + 3 * d * d + 2 * dm * dm
    for l in range(nl):
        base = l * ssz
        for j in range(d * d):
            st[base + d + j] *= factor                       # SWa
            st[base + d + d * d + j] *= factor               # SB
            st[base + d + 2 * d * d + j] *= factor           # SK
        for i in range(d):
            st[base + d + 3 * d * d + i] *= factor           # Sba
