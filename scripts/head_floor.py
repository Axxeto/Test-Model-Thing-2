"""U5 — LANTAI KEPALA SAJA: berapa CE terbaik yang bisa dicapai kepala output
`logits = W_out @ l2norm(y_top @ P[(Ltop,1)]) + byte_bias` TANPA batas dari SSM?

REVISI 3 (perbaikan A1 dari telaah eksternal):
  * PEMASANGAN benar. Versi lama memasangkan representasi `y_t` dengan byte
    MASUKAN `b_t` (mendekode token yang baru saja dibaca) dan TIDAK melewati
    kepala prediksi `P`. Yang terukur karena itu bukan tugas model. Sekarang
    pasangan yang benar adalah `X[t] -> b_{t+1}` (prekuensial), lewat `P[(Ltop,1)]`
    persis seperti `_stack_step` di jalur cepat.
  * TIGA KONTROL wajib (agar angka bisa ditafsirkan):
      (K1) X = onehot(b_{t+1})  -> harus ~0 bit  (kepala MAMPU memetakan token)
      (K2) X = noise(k)         -> harus ~log2 V (kepala TIDAK mengarang)
      (K3) X = onehot(b_t)      -> harus ~H(b_{t+1}|b_t) empiris (batas order-1)
  * EVAL HANGAT: `--eval_warm` melakukan burn-in tanpa belajar sebelum mengukur,
    supaya state lapisan lambat (half-life sampai ~1717 byte) tidak dibaca "dingin".

Tiga varian (seperti sebelumnya):
  (A) kepala acak            : CE acak (harus ~log2 V bila fitur tak informatif)
  (B) kepala dilatih, X beku : lantai kapasitas kepala
  (C) model penuh            : referensi
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tmt2.model import Config, TMT2
from tmt2.learner import Stream, TrainOpts
from tmt2.fast_learner import FastLearner
from tmt2.corpus import random_bytes

LN2 = float(np.log(2))


def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def ce_of(logits, b):
    m = logits.max()
    return float(m + np.log(np.exp(logits - m).sum()) - logits[b])


def l2n_rows(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


def collect_features(model, ln, data, warm=0):
    """Kumpulkan (l2norm(y_top @ P1), b_{t+1}) — PASANGAN PREKUENSIAL YANG BENAR.

    y_top pada langkah t memprediksi byte t+1. `P1 = P[(Ltop,1)]` adalah kepala
    horizon-1 lapisan puncak, persis seperti `_stack_step`. Baris terakhir tidak
    punya target (b_{t+1} tidak ada) jadi dibuang.
    """
    Ltop = model.cfg.n_layers - 1
    P1 = model.P[(Ltop, 1)].data
    # burn-in hangat (tanpa belajar) supaya state tidak dingin
    s = Stream(model, online=False)
    for b in data[:warm]:
        ln.step(s, int(b), train=False, think=False)
    X, Y = [], []
    for t, b in enumerate(data):
        ln.step(s, int(b), train=False, think=False)
        e = s.buf.get(s.t)
        if e is None:
            continue
        rec = e.chain[0][-1]                       # rekaman lapisan puncak
        y = np.asarray(rec.y)                      # y_top pada langkah t
        p1 = l2n_rows((y @ P1)[None, :])[0]        # PENTING: lewat P, lalu l2norm
        X.append(p1)
        if t + 1 < len(data):
            Y.append(int(data[t + 1]))             # target = byte BERIKUTNYA
    n = min(len(X), len(Y))
    return np.asarray(X[:n]), np.asarray(Y[:n])


def fit_head(Xn, Y, V, d, epochs, lr, rng):
    """Latih kepala linear `W @ x + b` dengan SGD mini-batch VEKTORISASI.

    Versi per-sampel (loop Python) ~100x lebih lambat dan membuat skrip ini tidak
    selesai dalam batas waktu wajar (3 kontrol x 3000 epoch x 512 sampel). Logits
    dan gradien dihitung sebagai matriks (B, V) sekaligus.
    """
    W = rng.normal(0, 0.02, (V, d))
    bias = np.zeros(V)
    n = len(Y)
    Yb = np.asarray(Y)
    B = 64
    for _ in range(epochs):
        idx = rng.permutation(n)
        for s0 in range(0, n, B):
            j = idx[s0:s0 + B]
            Xb = Xn[j]                                  # (B, d)
            Z = Xb @ W.T + bias                         # (B, V)
            Z -= Z.max(1, keepdims=True)
            E = np.exp(Z)
            P = E / E.sum(1, keepdims=True)             # (B, V)
            P[np.arange(len(j)), Yb[j]] -= 1.0          # gradien CE
            W -= lr * (P.T @ Xb) / len(j)
            bias -= lr * P.sum(0) / len(j)
    Z = Xn @ W.T + bias
    Z -= Z.max(1, keepdims=True)
    ce = float(np.mean(-Z[np.arange(n), Yb] + np.log(np.exp(Z).sum(1))) / LN2)
    return W, bias, ce


def emp_H(vals, V):
    c = np.bincount(vals, minlength=V).astype(float)
    p = c / c.sum()
    return float(-np.sum(p * np.log2(p + 1e-12)))


def emp_H2(pairs, V):
    """H(b_{t+1} | b_t) empiris."""
    tab = {}
    for a, b in pairs:
        tab.setdefault(a, np.zeros(V))[b] += 1
    tot, n = 0.0, len(pairs)
    for a, c in tab.items():
        p = c / c.sum()
        tot += (c.sum() / n) * float(-np.sum(p * np.log2(p + 1e-12)))
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--vocab", type=int, default=64)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--d_mem", type=int, default=8)
    ap.add_argument("--L", type=int, default=1)
    ap.add_argument("--deep", type=int, default=0)
    ap.add_argument("--se", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--head_epochs", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--head_lr", type=float, default=5e-2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--warm", type=int, default=0,
                    help="burn-in tanpa belajar sebelum mengukur (eval hangat)")
    ap.add_argument("--period", type=int, default=8)
    ap.add_argument("--frozen_features", action="store_true",
                    help="pakai representasi X dari model ACAK (tanpa latih)")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    data = random_bytes(a.n, a.vocab, a.seed)

    cfg = Config(d=a.d, n_layers=a.L, d_mem=a.d_mem, head="linear", deep_grad=bool(a.deep),
                 lr=a.lr, warmup=200, slow_every=a.se, clip=1.0, wd=0.0, replay=False,
                 adapter_rank=0, use_innovation=False, mem_query_x=True, mem_wq_match_k=True,
                 mem_hetero=True, mem_surprise_fixed=True, beta_max=1.0, seed=a.seed)
    m = TMT2(cfg)
    ln = FastLearner(m, TrainOpts(seed=a.seed, think_prob=0.0, use_stop=False, only_byte_loss=True))
    if not a.frozen_features:
        s = Stream(m)
        for _ in range(a.epochs):
            for b in data:
                ln.step(s, int(b))

    target = [int(b) for b in data]

    # ---------- (C) model penuh, EVAL KONSISTEN-HANGAT ----------
    def warm_ce():
        s = Stream(m, online=False)
        warm = min(a.warm, len(target) - 1)
        for b in target[:warm]:
            ln.step(s, int(b), train=False, think=False)
        out = []
        for b in target[warm:]:
            c = ln.step(s, int(b), train=False, think=False)
            if c is not None:
                out.append(c)
        return np.array(out)

    full_ce = float(warm_ce().mean() / LN2)
    if a.warm:
        cold = float(ln.eval_stream(target).mean() / LN2)
        print(f"eval dingin (tanpa burn-in)   : {cold:.4f} bit")
        print(f"eval HANGAT (burn-in {a.warm:>3}) : {full_ce:.4f} bit   (selisih {cold-full_ce:+.4f})")
    else:
        print(f"eval dingin                   : {full_ce:.4f} bit")

    # ---------- fitur X (pasangan benar) ----------
    X, Y = collect_features(m, ln, data, warm=a.warm)
    print(f"\nn={a.n} V={a.vocab} L={a.L} deep={a.deep} se={a.se} epochs={a.epochs} "
          f"frozen={a.frozen_features} | pasangan X[t]->b[t+1] = {len(Y)}")
    print(f"target entropi H_0 empiris    : {emp_H(Y, a.vocab):.4f} bit")
    print(f"target H_1 = H(b1|b0) empiris : "
          f"{emp_H2([(int(data[t]), int(data[t+1])) for t in range(len(data)-1)], a.vocab):.4f} bit")

    # ---------- TIGA KONTROL (dijalankan pada kepala yang SAMA arsitekturnya) ----------
    print("\nKONTROL (X yang DIKONSTRUKSI, latih kepala 3000 ep):")
    ctrls = {
        "K1 onehot(b_{t+1}) -> ~0":      np.eye(a.vocab)[Y],
        "K2 noise -> ~log2 V":           rng.normal(0, 1, (len(Y), a.vocab)),
        "K3 onehot(b_t) -> ~H1":         np.eye(a.vocab)[[int(b) for b in data[:len(Y)]]],
    }
    for name, Z in ctrls.items():
        Zn = l2n_rows(Z)
        _, _, ce = fit_head(Zn, Y, a.vocab, Zn.shape[1], a.head_epochs, a.head_lr,
                            np.random.default_rng(a.seed + 1))
        print(f"  {name:<28}: {ce:.4f} bit")

    # ---------- fitur model ----------
    Xn = l2n_rows(X)
    W = rng.normal(0, 0.02, (a.vocab, a.d))
    bias = np.zeros(a.vocab)
    rng_ce = float(np.mean([ce_of(W @ Xn[i] + bias, Y[i]) for i in range(len(Y))]) / LN2)
    t0 = time.time()
    W, bias, head_ce = fit_head(Xn, Y, a.vocab, a.d, a.head_epochs, a.head_lr, rng)
    uniq = len(np.unique([tuple(np.round(v, 6)) for v in Xn], axis=0))

    print("\nHASIL UTAMA:")
    print(f"  model penuh (eval konsisten) : {full_ce:.4f} bit")
    print(f"  kepala ACAK (tanpa latih)    : {rng_ce:.4f} bit")
    print(f"  kepala DILATIH, X beku       : {head_ce:.4f} bit   ({time.time()-t0:.1f}s)")
    print(f"  representasi X unik          : {uniq} / {len(Y)}")

    # ---------- CE per fase ----------
    P = a.period
    ev = warm_ce() / LN2
    ph_full = np.array([np.mean([ev[j] for j in range(len(ev)) if (j + 1) % P == k]) for k in range(P)])
    ce_head = np.array([ce_of(W @ Xn[i] + bias, Y[i]) for i in range(len(Y))]) / LN2
    ph_head = np.array([np.mean([ce_head[i] for i in range(len(ce_head)) if (i + 1) % P == k])
                        for k in range(P)])
    print(f"\nCE per fase (k mod {P}):")
    print("  fase        : " + " ".join(f"{k:>7}" for k in range(P)))
    print("  model penuh : " + " ".join(f"{v:>7.3f}" for v in ph_full))
    print("  kepala saja : " + " ".join(f"{v:>7.3f}" for v in ph_head))
    print(f"  selisih penuh={ph_full.max()-ph_full.min():.3f}  "
          f"kepala={ph_head.max()-ph_head.min():.3f}")

    print("\nINTERPRETASI (hanya sah bila K1~0 dan K2~log2 V):")
    if uniq < len(Y) * 0.5:
        print(f"  * Kolisi berat ({uniq}/{len(Y)}) -> masalah = EKSPRESIVITAS p.")
    elif head_ce < 0.5 and full_ce > head_ce:
        print(f"  * Kepala dapat {head_ce:.3f} bit tapi model penuh {full_ce:.3f} -> "
              "latihan tumpukan MERUSAK representasi (Temuan #3).")
    elif head_ce > 0.5:
        print(f"  * Kepala mentok {head_ce:.3f} bit -> KAPASITAS KEPALA.")
    else:
        print("  * Tidak ada hambatan jelas; cek kontrol K1/K2 dulu.")


if __name__ == "__main__":
    main()
