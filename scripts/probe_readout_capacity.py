"""PROBE KAPASITAS READOUT — apakah yang membatasi state SSM atau kapasitas readout?

`probe_state_info.py` memakai kepala LINEAR. Di sini tiga kapasitas readout diuji pada fitur
yang SAMA, semuanya DILATIH pada 70% pertama bagian tahan dan DINILAI pada 30% terakhir:
  linear : logits = W @ l2norm(f) + b            (kepala model sebenarnya)
  glu    : logits = W2 @ (silu(W1 x) * (V1 x))   (arsitektur readout model)
  mlp    : logits = W2 @ silu(W1 x) + b          (2 lapis, tersembunyi H)

Plus kontrol: seberapa besar konteks lokal eksplisit (3 byte sebelum byte sekarang, one-hot)
menambah DI ATAS state. Kalau MLP pada state tetap mentok, yang membatasi adalah KANDUNGAN
STATE (bukan readout) -> Gate 2 butuh rekurensi yang lebih kaya, bukan kepala.

⚠️ PELAJARAN (bug yang ditemukan & diperbaiki 2026-10-04): versi pertama mengevaluasi pada
himpunan LATIH, sehingga kepala berkapasitas tinggi tampak mencapai 0,34 bit (mustahil untuk
WikiText) — itu hafalan, bukan generalisasi. SELALU pisahkan latih/uji untuk probe.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from tmt2.model import Config, TMT2
from tmt2.corpus import load
from tmt2.learner import Stream, TrainOpts
from tmt2.fast_learner import FastLearner
from probe_state_info import collect, l2n_rows, fit_head

LN2 = float(np.log(2))


def _silu(z):
    return z / (1.0 + np.exp(-z))


def _silu_grad(z):
    s = 1.0 / (1.0 + np.exp(-z))
    return s + z * s * (1.0 - s)


def _ce(Z, Y):
    Zc = Z - Z.max(1, keepdims=True)
    return float(np.mean(Zc.max(1) + np.log(np.exp(Zc).sum(1))
                         - Zc[np.arange(len(Y)), Y]) / LN2)


def _adam(params, grads, state, lr, t, wd=0.0, b1=0.9, b2=0.999, eps=1e-8):
    for k, (p, g) in enumerate(zip(params, grads)):
        if wd:
            g = g + wd * p
        m, v = state[k]
        m *= b1
        m += (1 - b1) * g
        v *= b2
        v += (1 - b2) * g * g
        p -= lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)


def fit_mlp(Xtr, Ytr, Xte, Yte, V, H, epochs, lr, rng, batch=64, wd=1e-4):
    n, d = Xtr.shape
    W1 = rng.normal(0, np.sqrt(2.0 / d), (d, H))
    b1 = np.zeros(H)
    W2 = rng.normal(0, 0.02, (H, V))
    b2 = np.zeros(V)
    params = [W1, b1, W2, b2]
    st = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        for s0 in range(0, n, batch):
            j = rng.permutation(n)[s0:s0 + batch]
            xb = Xtr[j]
            z1 = xb @ W1 + b1
            h1 = _silu(z1)
            Z = h1 @ W2 + b2
            Zc = Z - Z.max(1, keepdims=True)
            E = np.exp(Zc)
            P = E / E.sum(1, keepdims=True)
            P[np.arange(len(j)), Ytr[j]] -= 1.0
            P /= len(j)
            t += 1
            gz1 = (P @ W2.T) * _silu_grad(z1)
            _adam(params, [xb.T @ gz1, gz1.sum(0), h1.T @ P, P.sum(0)], st, lr, t, wd)
    return _ce(_silu(Xte @ W1 + b1) @ W2 + b2, Yte)


def fit_glu(Xtr, Ytr, Xte, Yte, V, H, epochs, lr, rng, batch=64, wd=1e-4):
    n, d = Xtr.shape
    W1 = rng.normal(0, np.sqrt(2.0 / d), (d, H))
    b1 = np.zeros(H)
    V1 = rng.normal(0, np.sqrt(2.0 / d), (d, H))
    c1 = np.zeros(H)
    W2 = rng.normal(0, 0.02, (H, V))
    b2 = np.zeros(V)
    params = [W1, b1, V1, c1, W2, b2]
    st = [(np.zeros_like(p), np.zeros_like(p)) for p in params]
    t = 0
    for _ in range(epochs):
        for s0 in range(0, n, batch):
            j = rng.permutation(n)[s0:s0 + batch]
            xb = Xtr[j]
            z1 = xb @ W1 + b1
            g1 = xb @ V1 + c1
            h1 = _silu(z1) * g1
            Z = h1 @ W2 + b2
            Zc = Z - Z.max(1, keepdims=True)
            E = np.exp(Zc)
            P = E / E.sum(1, keepdims=True)
            P[np.arange(len(j)), Ytr[j]] -= 1.0
            P /= len(j)
            t += 1
            gh1 = P @ W2.T
            gz1 = gh1 * g1 * _silu_grad(z1)
            gg1 = gh1 * _silu(z1)
            _adam(params, [xb.T @ gz1, gz1.sum(0), xb.T @ gg1, gg1.sum(0),
                           h1.T @ P, P.sum(0)], st, lr, t, wd)
    return _ce((_silu(Xte @ W1 + b1) * (Xte @ V1 + c1)) @ W2 + b2, Yte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="wikitext")
    ap.add_argument("--ntrain", type=int, default=200000)
    ap.add_argument("--nheld", type=int, default=10000)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    a = ap.parse_args()

    raw = np.frombuffer(load(a.data), dtype=np.uint8)
    tr, held = raw[:a.ntrain], raw[a.ntrain:a.ntrain + a.nheld]
    cfg = Config.best(d=a.d, n_layers=a.L, vocab=256, d_mem=max(8, a.d // 4), use_memory=True,
                      head="linear", deep_grad=True, adapter_rank=0, mixer_k=0,
                      think_max=0, use_innovation=False, seg_len=10 ** 9, seed=a.seed)
    m = TMT2(cfg)
    L = FastLearner(m, TrainOpts(only_byte_loss=True, use_fused_kernel=True, seed=a.seed,
                                 use_stop=False))
    s = Stream(m, online=True)
    for b in tr:
        L.step_fused(s, int(b), train=True)
    print(f"data={a.data} ntrain={len(tr)} nheld={len(held)} d={a.d} L={a.L} seed={a.seed}",
          flush=True)
    print(f"CE model sendiri (tahan): "
          f"{float(L.eval_stream([int(b) for b in held]).mean()/LN2):.4f} bit", flush=True)

    feats, Y, CTX = collect(m, L, held, kctx=4)
    n = len(Y)
    ntr = int(0.7 * n)
    print(f"pasangan={n} (latih kepala={ntr}, uji={n-ntr})", flush=True)

    hist3 = np.zeros((n, 3 * 256))
    for i, c in enumerate(CTX):
        for k in range(3):
            idx = -(k + 2)                       # b_{t-1}, b_{t-2}, b_{t-3}
            if len(c) >= -idx:
                hist3[i, k * 256 + int(c[idx])] = 1.0

    featset = {
        "E[b_t]": feats["E"],
        f"h{a.L-1} (state)": feats[f"h{a.L-1}"],
        f"q{a.L-1} (masukan kepala)": feats[f"q{a.L-1}"],
        "h+3byte (state+lokal)": np.concatenate([feats[f"h{a.L-1}"], hist3], axis=1),
    }

    hdr = f"{'fitur':<26}{'dim':>6}{'linear':>10}{'glu':>10}{'mlp':>10}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, F in featset.items():
        X = l2n_rows(F)
        Xtr, Ytr, Xte, Yte = X[:ntr], Y[:ntr], X[ntr:], Y[ntr:]
        ce_lin = fit_head(Xtr, Ytr, 256, 800, 0.05, np.random.default_rng(a.seed + 7), Xte, Yte)
        ce_glu = fit_glu(Xtr, Ytr, Xte, Yte, 256, a.H, a.epochs, a.lr,
                         np.random.default_rng(a.seed + 11), wd=a.wd)
        ce_mlp = fit_mlp(Xtr, Ytr, Xte, Yte, 256, a.H, a.epochs, a.lr,
                         np.random.default_rng(a.seed + 13), wd=a.wd)
        print(f"{name:<26}{X.shape[1]:>6}{ce_lin:>10.4f}{ce_glu:>10.4f}{ce_mlp:>10.4f}",
              flush=True)

    print("\n== bacaan ==")
    print("  mlp(state) ~ linear(state) -> KANDUNGAN STATE yang membatasi (bukan readout).")
    print("  'h+3byte' jauh lebih baik -> konteks lokal eksplisit memang yang hilang.")


if __name__ == "__main__":
    main()
