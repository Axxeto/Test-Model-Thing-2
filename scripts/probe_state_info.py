"""PROBE INFO-STATE — di mana informasi byte-berikutnya hilang?

Pertanyaan (STATUS §1 "arah yang belum dijelajahi"): apakah hambatan ada di
REKURENSI SSM (state h tidak memuat info) atau di JALUR BACA non-memori
(Wc/GLU/kepala membuang info)?

Prosedur:
  1. Latih model pada data[:NTRAIN] (jalur fused, cepat). Bekukan.
  2. Lintasi data[NTRAIN:NTRAIN+NHELD] TANPA belajar; pada tiap langkah t rekam
     beberapa representasi yang SEMUANYA berada di jalur prekuensial:
        E[b_t]        embedding byte sekarang         (kontrol bigram)
        h_l           state SSM lapisan l              (isyarat rekurensi)
        q_l           l2norm(y_l @ P[(l,1)])           (masukan kepala prediksi lapisan l)
        p_top         q_{L-1}                          (masukan kepala byte SESUNGGUHNYA)
     Target = byte berikutnya b_{t+1}.
  3. Latih kepala LINEAR yang sama (logits = W @ l2norm(f) + b) pada 70% pertama
     bagian tahan, uji pada 30% terakhir.
  4. Bandingkan dengan n-gram empiris yang DILATIH pada korpus latih dan diuji
     pada bagian uji YANG SAMA (out-of-sample, adil).

Tafsir:
  * E[b_t] ~ order-1  -> kepala mampu (kontrol).
  * h_l << E[b_t]     -> rekurensi menambah info (bagus).
  * q_top > h_l       -> jalur baca MEMBUANG info (kepala/Wc/GLU).
  * q_top ~ E[b_t]    -> seluruh model tak lebih baik dari bigram.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from tmt2.model import Config, TMT2
from tmt2.corpus import load
from tmt2.learner import Stream, TrainOpts
from tmt2.fast_learner import FastLearner

LN2 = float(np.log(2))


def l2n_rows(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (n + 1e-8)


def fit_head(X, Y, V, epochs, lr, rng, Xte=None, Yte=None):
    """Kepala linear `logits = W @ x + b`, SGD mini-batch tervektorisasi. X sudah l2norm.

    ⚠️ PENTING: CE yang dikembalikan dihitung pada (Xte, Yte) bila diberikan — BUKAN pada
    data latih. Versi lama mengevaluasi pada himpunan latih sehingga angkanya adalah CE LATIH
    (terlalu optimistis; pada kepala berkapasitas tinggi bisa tampak ~0,3 bit padahal mustahil).
    """
    n, d = X.shape
    W = rng.normal(0, 0.02, (V, d))
    b = np.zeros(V)
    B = 64
    for _ in range(epochs):
        idx = rng.permutation(n)
        for s0 in range(0, n, B):
            j = idx[s0:s0 + B]
            Z = X[j] @ W.T + b
            Z -= Z.max(1, keepdims=True)
            E = np.exp(Z)
            P = E / E.sum(1, keepdims=True)
            P[np.arange(len(j)), Y[j]] -= 1.0
            W -= lr * (P.T @ X[j]) / len(j)
            b -= lr * P.sum(0) / len(j)
    Xe = X if Xte is None else Xte
    Ye = Y if Yte is None else Yte
    Z = Xe @ W.T + b
    Zc = Z - Z.max(1, keepdims=True)
    return float(np.mean(Zc.max(1) + np.log(np.exp(Zc).sum(1))
                         - Zc[np.arange(len(Ye)), Ye]) / LN2)


def ngram_ce(train, ctxs, curs, V, order, lam=0.5):
    """N-gram interpolasi Jelinek-Mercer: p_o = lam*p_ML(o) + (1-lam)*p_{o-1}.

    Tabel dari `train`; diuji pada (ctxs, curs) yang out-of-sample. `ctx` = byte SEBELUM
    target, terurut lama->baru, jadi konteks order-o = ctx[-o:].
    """
    train = np.asarray(train)
    uni = np.bincount(train, minlength=V).astype(float)
    uni = (uni + 1e-9) / (uni.sum() + 1e-9 * V)
    tabs, tots = [], []
    for o in range(1, order + 1):
        tab, tot = {}, {}
        for i in range(o, len(train)):
            c = tuple(int(x) for x in train[i - o:i])
            b = int(train[i])
            tab.setdefault(c, np.zeros(V))[b] += 1.0
            tot[c] = tot.get(c, 0.0) + 1.0
        tabs.append(tab)
        tots.append(tot)
    ce = 0.0
    for ctx, b in zip(ctxs, curs):
        ctx = tuple(int(x) for x in ctx)
        p = uni
        for o in range(1, order + 1):
            c = ctx[-o:]
            if len(c) < o:
                continue
            cc = tabs[o - 1].get(c)
            if cc is None:
                continue
            p = lam * (cc / tots[o - 1][c]) + (1 - lam) * p
        ce += -np.log2(p[b] + 1e-12)
    return ce / len(curs)


def collect(m, L, data, kctx=3):
    """Kumpulkan fitur per langkah t (setelah memproses b_t) + target b_{t+1}."""
    nl = m.cfg.n_layers
    s = Stream(m, online=False)
    keys = ["E"] + [f"h{l}" for l in range(nl)] + [f"q{l}" for l in range(nl)]
    acc = {k: [] for k in keys}
    ytarget, ctx = [], []
    hist = []
    for i, b in enumerate(data):
        b = int(b)
        L.step(s, b, train=False, think=False)
        e = s.buf.get(s.t)
        hist.append(b)
        if e is None or i + 1 >= len(data):
            continue
        acc["E"].append(m.encode_byte(b).data.copy())
        for l in range(nl):
            yl = np.asarray(e.chain[e.k][l].y.data)
            acc[f"h{l}"].append(s.st.layers[l].h.copy())
            key = (l, 1)
            acc[f"q{l}"].append(yl @ m.P[key].data if key in m.P else yl)
        ytarget.append(int(data[i + 1]))
        ctx.append(tuple(hist[-kctx:]))                  # kctx byte terakhir (termasuk b_t)
    return ({k: np.asarray(v, float) for k, v in acc.items()},
            np.asarray(ytarget), np.asarray(ctx, dtype=object))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="wikitext")
    ap.add_argument("--ntrain", type=int, default=200000)
    ap.add_argument("--nheld", type=int, default=12000)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=1500, help="epoch kepala probe")
    ap.add_argument("--head_lr", type=float, default=0.05)
    ap.add_argument("--split", type=float, default=0.7)
    a = ap.parse_args()

    raw = np.frombuffer(load(a.data), dtype=np.uint8)
    V = 256
    tr = raw[:a.ntrain]
    held = raw[a.ntrain:a.ntrain + a.nheld]
    print(f"data={a.data} ntrain={len(tr)} nheld={len(held)} d={a.d} L={a.L} seed={a.seed}")

    cfg = Config.best(d=a.d, n_layers=a.L, vocab=V, d_mem=max(8, a.d // 4), use_memory=True,
                      head="linear", deep_grad=True, adapter_rank=0, mixer_k=0,
                      think_max=0, use_innovation=False, seg_len=10 ** 9, seed=a.seed)
    m = TMT2(cfg)
    o = TrainOpts(only_byte_loss=True, use_fused_kernel=True, seed=a.seed, use_stop=False)
    L = FastLearner(m, o)
    s = Stream(m, online=True)
    t0 = time.time()
    for i in range(len(tr)):
        L.step_fused(s, int(tr[i]), train=True)
    dt = time.time() - t0
    print(f"latih {len(tr)} byte dalam {dt:.1f}s ({len(tr)/dt:.0f} B/s)")

    own = L.eval_stream([int(b) for b in held])
    print(f"CE model sendiri (tahan)      : {float(own.mean()/LN2):.4f} bit")

    feats, Y, CTX = collect(m, L, held)
    n = len(Y)
    ntr = int(a.split * n)
    print(f"pasangan terkumpul={n} (latih kepala={ntr}, uji={n-ntr})")

    Yte, CTXte = Y[ntr:], CTX[ntr:]
    print("\nbaseline n-gram (tabel dari korpus LATIH, diuji pada bagian uji):")
    print(f"  order-0 : {ngram_ce(tr, [()] * len(Yte), Yte, V, 0):.4f} bit")
    print(f"  order-1 : {ngram_ce(tr, [c[-1:] for c in CTXte], Yte, V, 1):.4f} bit")
    print(f"  order-2 : {ngram_ce(tr, [c[-2:] for c in CTXte], Yte, V, 2):.4f} bit")
    print(f"  order-3 : {ngram_ce(tr, [c[-3:] for c in CTXte], Yte, V, 3):.4f} bit")

    print(f"\n{'representasi':<12}{'dim':>5}{'CE probe':>12}")
    print("-" * 29)
    rows = {}
    for k in ["E"] + [f"h{l}" for l in range(a.L)] + [f"q{l}" for l in range(a.L)]:
        X = l2n_rows(feats[k])
        rng = np.random.default_rng(a.seed + 7)
        ce = fit_head(X[:ntr], Y[:ntr], V, a.epochs, a.head_lr, rng, X[ntr:], Y[ntr:])
        rows[k] = ce
        print(f"{k:<12}{X.shape[1]:>5}{ce:>12.4f}", flush=True)

    top = f"h{a.L-1}"
    if "E" in feats and top in feats:
        X = l2n_rows(np.concatenate([feats["E"], feats[top]], axis=1))
        rng = np.random.default_rng(a.seed + 7)
        rows["E+" + top] = fit_head(X[:ntr], Y[:ntr], V, a.epochs, a.head_lr, rng,
                                    X[ntr:], Y[ntr:])
        print(f"{'E+' + top:<12}{X.shape[1]:>5}{rows['E+' + top]:>12.4f}", flush=True)

    print("\n== bacaan ==")
    hbest = min(rows.get(f"h{l}", 9) for l in range(a.L))
    print(f"  byte sekarang E[b_t] : {rows['E']:.3f} bit")
    print(f"  state SSM terbaik    : {hbest:.3f} bit")
    print(f"  kepala model q_top   : {rows.get('q' + str(a.L - 1), float('nan')):.3f} bit")
    print(f"  -> rekurensi menambah {rows['E'] - hbest:+.3f} bit di atas byte sekarang")
    print(f"  -> jalur baca mengubah {hbest - rows.get('q' + str(a.L - 1), 9):+.3f} bit dari state terbaik")


if __name__ == "__main__":
    main()
