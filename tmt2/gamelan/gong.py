"""Gong — rekurensi linear dengan peluruhan bergantung-data, bentuk REKUREN dan CHUNKWISE.

  U_t = λ_t ⊙ U_{t-1} + (1 − λ_t) k_t v_tᵀ        (state per head, O(1) per langkah)
  o_t = q_t U_t

Dua bentuk yang HARUS identik (tes paritas #1, ARCHITECTURE_V2.md §7):

  * **rekuren** — dipakai saat inferensi/generasi; satu langkah O(D²) per head.
  * **chunkwise** — dipakai saat latihan; di dalam chunk menjadi matmul (BLAS), state dibawa
    antar-chunk. Inilah yang memecah plafon kecepatan RTRL: tidak ada jejak `3·d²` per byte.

Catatan numerik penting: peluruhan relatif di dalam chunk dihitung dengan **produk berjalan**,
BUKAN dengan membagi produk kumulatif (`cum[i]/cum[j]`). Pembagian itu meledak/underflow begitu
ada λ kecil, dan chunk panjang dengan λ kecil adalah kasus normal.
"""
import numpy as np


def gong_recurrent(q, k, v, lam, U0):
    """Bentuk rekuren (acuan kebenaran).

    q, k: (T, H, Dk) · v: (T, H, Dv) · lam: (T, H) · U0: (H, Dk, Dv)
    Mengembalikan (o (T,H,Dv), U_akhir (H,Dk,Dv)).
    """
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)
    T, H, Dk = q.shape
    Dv = v.shape[2]
    U = np.array(U0, dtype=np.float64, copy=True)
    o = np.empty((T, H, Dv), dtype=np.float64)
    for t in range(T):
        for h in range(H):
            l = lam[t, h]
            U[h] = l * U[h] + (1.0 - l) * np.outer(k[t, h], v[t, h])
            o[t, h] = q[t, h] @ U[h]
    return o, U


def gong_chunkwise(q, k, v, lam, U0, chunk=64):
    """Bentuk chunkwise TERVEKTORISASI (matmul/BLAS). Harus sama dengan `gong_recurrent`.

    Di dalam chunk:
      o_i = cum_i · (q_i U0)                       <- kontribusi antar-chunk
          + Σ_{j≤i} rel_{j→i} (1−λ_j) (q_i·k_j) v_j <- kontribusi intra-chunk (matmul)

    Peluruhan relatif dihitung di ruang log: ``rel_{j→i} = exp(cumlog_i − cumlog_j)``.
    Ini stabil — begitu peluruhan sangat besar, exp-nya underflow ke 0, dan itu memang benar.
    """
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    lam = np.asarray(lam, dtype=np.float64)
    T, H, Dk = q.shape
    Dv = v.shape[2]
    U = np.array(U0, dtype=np.float64, copy=True)
    o = np.empty((T, H, Dv), dtype=np.float64)
    tril = None
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        n = e - s
        qc, kc, vc, lc = q[s:e], k[s:e], v[s:e], lam[s:e]
        if tril is None or tril.shape[0] != n:
            tril = np.tril(np.ones((n, n), dtype=bool))
        for h in range(H):
            lh = lc[:, h]
            cl = np.cumsum(np.log(np.maximum(lh, 1e-300)))     # log produk kumulatif
            cum = np.exp(cl)
            o[s:e, h] = cum[:, None] * (qc[:, h] @ U[h])       # antar-chunk
            rel = np.exp(cl[:, None] - cl[None, :])            # rel[i,j] = Π_{j<r≤i} λ_r
            D = np.where(tril, rel * (1.0 - lh)[None, :], 0.0)
            S = (qc[:, h] @ kc[:, h].T) * D                    # (n,n) <- matmul
            o[s:e, h] += S @ vc[:, h]                          # (n,Dv) <- matmul
            U[h] = cum[-1] * U[h] + np.einsum('j,jd,je->de', D[-1], kc[:, h], vc[:, h])
    return o, U
