"""TINGKAT B (neural paralel-chunk) — latih mandiri pada korpus Indonesia, uji BPB & generasi.

Tingkat B = Saron (konv kausal) + Gong (rekurensi linear peluruhan data-dependen) + gerbang +
SwiGLU + RMSNorm, kepala `logits = W_out · norm(y)`. Dilatih teacher-forced pada byte CE,
BPTT penuh di dalam deret (bentuk rekuren; gradien identik dengan bentuk chunkwise).

Protokol (konsisten dengan proyek):
  * jendela uji = EKOR tetap SETELAH prefix latih (tak beririsan);
  * target = byte berikutnya (x[t] memprediksi x[t+1]); posisi terakhir tanpa target;
  * BPB dihitung per BYTE (bukan per bit).

Jalan:
  python scripts/tierb_train.py --corpus data/indonesian_1m_pure.txt --ntrain 250000 --epochs 3
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from tmt2.gamelan.block import GamelanBlock
from tmt2.gamelan.train import block_forward_cache, block_backward

LN2 = float(np.log(2))


class Adam:
    def __init__(self, params, lr):
        self.p = params
        self.lr = lr
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads):
        self.t += 1
        b1, b2, eps = 0.9, 0.999, 1e-8
        for k in self.p:
            g = grads.get(k)
            if g is None:
                continue
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            self.p[k] -= self.lr * mh / (np.sqrt(vh) + eps)


def make_params(blk, d, seed):
    rng = np.random.default_rng(seed)
    E = rng.normal(0, 1.0, (256, d))
    W_out = rng.normal(0, 1.0 / np.sqrt(d), (256, d))
    params = {"E": E, "W_out": W_out}
    for nm in ("wd", "bd", "wp", "wq", "wk", "wv", "wl", "bl", "wo", "wg", "bg",
               "w1", "w2", "g1", "g2"):
        params[nm] = getattr(blk, nm)
    return params, E, W_out


def train_tierB(data, d, heads, conv_k, epochs, lr, batch, seed, log=print):
    blk = GamelanBlock(d=d, heads=heads, conv_k=conv_k, seed=seed)
    params, E, W_out = make_params(blk, d, seed)
    opt = Adam(params, lr)
    N = len(data)
    t0 = time.perf_counter()
    for ep in range(epochs):
        tot, nb = 0.0, 0
        for s in range(0, N - batch - 1, batch):
            b = data[s:s + batch].astype(np.int64)
            tgt = data[s + 1:s + batch + 1].astype(np.int64)
            y, cache = block_forward_cache(blk, E[b])
            logits = y @ W_out.T
            lo = logits - logits.max(1, keepdims=True)
            Ex = np.exp(lo)
            P = Ex / Ex.sum(1, keepdims=True)
            tot += float(-np.mean(np.log(P[np.arange(len(b)), tgt] + 1e-300)))
            nb += 1
            dlog = P.copy()
            dlog[np.arange(len(b)), tgt] -= 1.0
            dlog /= len(b)
            dy = dlog @ W_out
            dx, g = block_backward(blk, cache, dy)
            grads = dict(g)
            grads["W_out"] = dlog.T @ y
            xgrad = np.zeros_like(E)
            np.add.at(xgrad, b, dx)
            grads["E"] = xgrad
            opt.step(grads)
        el = time.perf_counter() - t0
        log(f"    epoch {ep+1}/{epochs}: CE latih {tot/max(1,nb)/LN2:.4f} bit · "
            f"{N*epochs/max(el,1e-9):.0f} B/s efektif")
    return blk, E, W_out


def eval_bpb(blk, E, W_out, te, batch=4096):
    """BPB byte-berikutnya pada jendela uji (target = te[1:])."""
    N = len(te)
    nll = 0.0
    cnt = 0
    for s in range(0, N - 1, batch):
        e = min(s + batch, N - 1)
        b = te[s:e].astype(np.int64)
        y, _ = blk.forward(E[b], mode="chunk", chunk=128)
        logits = y @ W_out.T
        tgt = te[s + 1:e + 1].astype(np.int64)
        lo = logits - logits.max(1, keepdims=True)
        Ex = np.exp(lo)
        P = Ex / Ex.sum(1, keepdims=True)
        nll += float(-np.log2(P[np.arange(len(b)), tgt] + 1e-300).sum())
        cnt += len(b)
    return nll / max(cnt, 1)


def generate(blk, E, W_out, prompt, n, temp, seed):
    rng = np.random.default_rng(seed)
    if isinstance(prompt, str):
        prompt = prompt.encode("utf-8")
    state = blk.init_state()
    y = None
    for b in prompt:
        y, state = blk.step(E[int(b)], state)
    out = bytearray()
    for _ in range(int(n)):
        lg = (y @ W_out.T) / max(temp, 1e-6)
        lg = lg - lg.max()
        p = np.exp(lg)
        p /= p.sum()
        b = int(rng.choice(256, p=p))
        out.append(b)
        y, state = blk.step(E[b], state)
    return bytes(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/indonesian_1m_pure.txt")
    ap.add_argument("--ntrain", type=int, default=250000)
    ap.add_argument("--ntest", type=int, default=8000)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--conv_k", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gen", type=int, default=300)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--out", default="results/tierb_gen.txt")
    ap.add_argument("--jsonl", default="results/tierb_train.jsonl")
    a = ap.parse_args()

    with open(a.corpus, "rb") as f:
        raw = np.frombuffer(f.read(a.ntrain + a.ntest), dtype=np.uint8)
    total = raw.size
    if a.ntrain + a.ntest > total:
        raise SystemExit(f"korpus {total} B < {a.ntrain + a.ntest}")
    tr, te = raw[:a.ntrain], raw[a.ntrain:a.ntrain + a.ntest]

    print(f"KORPUS {a.corpus} · latih {a.ntrain:,} B · uji {a.ntest:,} B (ekor, tak beririsan)")
    print(f"Tingkat B · d={a.d} H={a.heads} conv_k={a.conv_k} · {a.epochs} epoch · "
          f"lr={a.lr} batch={a.batch}\n", flush=True)

    t0 = time.perf_counter()
    blk, E, W_out = train_tierB(tr, a.d, a.heads, a.conv_k, a.epochs, a.lr, a.batch, a.seed)
    tt = time.perf_counter() - t0
    bps = a.ntrain * a.epochs / max(tt, 1e-9)

    bpb = eval_bpb(blk, E, W_out, te)
    print(f"\nlatih selesai {tt:.0f}s · ~{bps:.0f} B/s · BPB tahan Tingkat B: {bpb:.4f}")

    if a.ckpt:
        os.makedirs(os.path.dirname(a.ckpt), exist_ok=True)
        np.savez_compressed(a.ckpt, E=E, W_out=W_out,
                            **{nm: getattr(blk, nm) for nm in
                               ("wd", "bd", "wp", "wq", "wk", "wv", "wl", "bl", "wo",
                                "wg", "bg", "w1", "w2", "g1", "g2")},
                            meta=np.array([json.dumps(dict(
                                corpus=a.corpus, ntrain=a.ntrain, ntest=a.ntest, bpb=bpb,
                                d=a.d, heads=a.heads, conv_k=a.conv_k,
                                epochs=a.epochs, lr=a.lr))], dtype=object))
        print(f"checkpoint -> {a.ckpt}")

    prompts = ["Indonesia adalah negara kepulauan yang",
               "Pada tahun 1945, para pendiri bangsa",
               "Kucing merupakan hewan mamalia yang"]
    lines = [f"# Generasi Tingkat B — {a.corpus} (latih {a.ntrain:,} B, {a.epochs} epoch)",
             f"# BPB tahan {bpb:.4f} · d={a.d} H={a.heads} conv_k={a.conv_k}", ""]
    for p in prompts:
        for temp in (0.6, 0.9):
            txt = generate(blk, E, W_out, p, a.gen, temp, seed=11)
            lines.append(f"=== prompt: {p!r}  (temp {temp}) ===")
            lines.append(p + txt.decode("utf-8", errors="replace"))
            lines.append("")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"generasi -> {a.out}")

    with open(a.jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(corpus=a.corpus, ntrain=a.ntrain, ntest=a.ntest, bpb=bpb,
                                bps=bps, d=a.d, heads=a.heads, conv_k=a.conv_k,
                                epochs=a.epochs, lr=a.lr, train_s=tt)) + "\n")

    print("\n---- cuplikan (prompt #1, temp 0.6) ----")
    print(prompts[0] + generate(blk, E, W_out, prompts[0], 300, 0.6, 11)
          .decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
