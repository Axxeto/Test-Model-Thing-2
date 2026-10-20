"""Uji skala besar: latih TMT2 pada N byte/token dan laporkan BPB + baseline + generasi.

  python scripts/bigtest.py --data wikitext --head linear --deep_grad --only_byte ...
  python scripts/bigtest.py --bpe --vocab 4096 ...        # tokenisasi subword (BPE)

Melaporkan:
  * BPB prekuensial per blok (untuk melihat turun / plateau) dan kumulatif
  * BPB akhir vs unigram dan n-gram adaptif order 1/2 (selalu pada BYTE yang sama)
  * contoh generasi
  * waktu eksekusi total dan throughput
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tmt2.model import Config, TMT2
from tmt2.learner import Stream, TrainOpts
from tmt2.fast_learner import FastLearner
from tmt2.corpus import load
from tmt2 import tokenizer as tk


def ngram_bits(data, order):
    """BPB prekuensial n-gram adaptif dengan back-off (pada byte)."""
    counts = [dict() for _ in range(order + 1)]
    out = []
    for i, b in enumerate(data):
        p = 1.0 / 256
        for o in range(order + 1):
            if i < o:
                break
            c = counts[o].get(bytes(data[i - o:i]))
            if c is None:
                c = counts[o][bytes(data[i - o:i])] = [np.zeros(256), 0.0]
            p = (c[0][b] + 2.0 * p) / (c[1] + 2.0)
        out.append(-np.log2(p))
        for o in range(order + 1):
            if i < o:
                break
            c = counts[o][bytes(data[i - o:i])]
            c[0][b] += 1
            c[1] += 1
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50000)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--block", type=int, default=5000, help="ukuran blok dalam BYTE")
    ap.add_argument("--slow_every", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=1, help=">1: ulangi n byte (uji overfit)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="license")
    ap.add_argument("--head", default="linear", choices=["cosine", "linear"],
                    help="kepala keluaran (default: linear = terbaik)")
    ap.add_argument("--deep_grad", action="store_true",
                    help="alirkan gradien loss byte ke seluruh tumpukan (rekomendasi: aktif)")
    ap.add_argument("--think_prob", type=float, default=0.0)
    ap.add_argument("--use_stop", action="store_true")
    ap.add_argument("--only_byte", action="store_true")
    ap.add_argument("--ngram_lam", type=float, default=0.0,
                    help=">0: tambahkan lam*log p_bigram_online ke logits (kepala residual)")
    ap.add_argument("--ngram_learn", action="store_true",
                    help="pelajari skala prior n-gram sebagai parameter ng_lam")
    ap.add_argument("--ngram_order", type=int, default=1, help="orde konteks prior (1=bigram)")
    ap.add_argument("--bpe", action="store_true", help="tokenisasi subword BPE")
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--bpe_cache", default=None)
    ap.add_argument("--tok_frac", type=float, default=1.0,
                    help="porsi korpus untuk MELATIH tokenizer (1.0 = seluruhnya)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--gen", type=int, default=3)
    ap.add_argument("--gen_len", type=int, default=200)
    ap.add_argument("--d_mem", type=int, default=None,
                    help="dimensi memori (default: max(8, d//4))")
    ap.add_argument("--mem_query_x", action="store_true", default=None,
                    help="kueri baca dari ln(x_t), bukan ln(h_t) (default Config: aktif)")
    ap.add_argument("--no_mem_query_x", dest="mem_query_x", action="store_false",
                    help="matikan mem_query_x (perilaku lama)")
    ap.add_argument("--mem_wq_match_k", action="store_true", default=None,
                    help="init Wq = Wk (selaraskan ruang baca & tulis; default Config: aktif)")
    ap.add_argument("--no_mem_wq_match_k", dest="mem_wq_match_k", action="store_false",
                    help="matikan mem_wq_match_k (Wq acak; perilaku lama)")
    ap.add_argument("--mem_hetero", action="store_true", default=None,
                    help="kunci tulis dari konteks sebelumnya (asosiasi hetero)")
    ap.add_argument("--mem_surprise_fixed", action="store_true", default=None,
                    help="beta = beta_max tetap (lepas dari gerbang kejutan)")
    ap.add_argument("--beta_max", type=float, default=None)
    ap.add_argument("--no_mem", action="store_true", help="matikan memori delta sepenuhnya")
    ap.add_argument("--no_gate", action="store_true",
                    help="matikan suku dL/dh.(dh/dx) pada g_x antar-lapisan (perilaku lama)")
    ap.add_argument("--resume", default=None, help="muat bobot dari checkpoint .npz lalu lanjut")
    ap.add_argument("--save_every", type=int, default=0,
                    help="simpan checkpoint tiap N token (untuk sesi panjang / Kaggle)")
    a = ap.parse_args()

    raw = load(a.data)[:a.n]
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    tok = None
    if a.bpe:
        cut = int(len(raw) * a.tok_frac)
        pref = "" if a.tok_frac >= 1.0 else f"_pref{int(a.tok_frac*100)}"
        cache = a.bpe_cache or os.path.join(root, "data", f"bpe_{a.data}_{a.vocab}{pref}.json")
        tok = tk.get(raw if a.tok_frac >= 1.0 else raw[:cut], a.vocab, cache)
        ids, blens, canon = tk.encode(tok, raw)
        data, vocab = ids, a.vocab
        print(f"BPE V={a.vocab}: tokenizer dilatih pada {min(cut, len(raw))} byte; "
              f"{len(canon)} byte -> {len(ids)} token ({len(canon)/max(len(ids),1):.2f} byte/token)  "
              f"cache={os.path.basename(cache)}", flush=True)
    else:
        canon = raw
        data = np.frombuffer(raw, np.uint8).astype(np.int64)
        blens = np.ones(len(data), dtype=np.int64)
        vocab = 256
    data = np.tile(data, a.epochs)
    blens = np.tile(blens, a.epochs)

    cfg = Config(d=a.d, n_layers=a.layers, vocab=vocab,
                 d_mem=(a.d_mem if a.d_mem else max(8, a.d // 4)), lr=a.lr,
                 warmup=a.warmup, slow_every=a.slow_every, seed=a.seed, head=a.head,
                 deep_grad=a.deep_grad, ngram_learn=a.ngram_learn,
                 use_memory=(not a.no_mem), deep_grad_gate=(not a.no_gate))
    # flag memori: hanya timpa Config bila pengguna menyebutkannya secara eksplisit
    for _name, _val in (("mem_query_x", a.mem_query_x), ("mem_wq_match_k", a.mem_wq_match_k),
                        ("mem_hetero", a.mem_hetero), ("mem_surprise_fixed", a.mem_surprise_fixed),
                        ("beta_max", a.beta_max)):
        if _val is not None:
            setattr(cfg, _name, _val)
    model = TMT2(cfg)
    model.logit_scale.data = np.array(np.log(3.0))
    opts = TrainOpts(seed=a.seed, think_prob=a.think_prob, use_stop=a.use_stop,
                     only_byte_loss=a.only_byte, ngram_lam=a.ngram_lam,
                     ngram_order=a.ngram_order)
    ln = FastLearner(model, opts)
    if a.resume:
        # lanjutkan dari checkpoint: bobot dipulihkan, aliran data mulai dari awal lagi
        ln = FastLearner.load(a.resume, opts)
        model = ln.m
        cfg = model.cfg
        print(f"resume: bobot dipulihkan dari {a.resume}", flush=True)
    s = Stream(model)
    print(f"FastLearner d={cfg.d} lapisan={cfg.n_layers} vocab={cfg.vocab} "
          f"slow_every={cfg.slow_every} head={cfg.head} deep_grad={cfg.deep_grad} "
          f"lr={cfg.lr} warmup={cfg.warmup} only_byte={a.only_byte}", flush=True)

    bits = np.full(len(data), np.nan)
    lams = [] if model.ng_lam is not None else None
    t0 = time.time()
    for i, b in enumerate(data):
        c = ln.step(s, int(b))
        if c is not None:
            bits[i] = c / np.log(2)
        if lams is not None:
            lams.append(float(model.ng_lam.data))
        if a.save_every and (i + 1) % a.save_every == 0:
            ln.save(a.out or "checkpoint.npz")
            print(f"  [ckpt] {i+1} token -> {a.out or 'checkpoint.npz'} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    dt = time.time() - t0
    cbytes = np.cumsum(blens)
    total_bytes = int(cbytes[-1])
    print(f"\nwaktu total: {dt:.1f} s   token/detik: {len(data)/dt:.0f}   "
          f"byte/detik: {total_bytes/dt:.0f} B/s", flush=True)

    print(f"\n== BPB prekuensial per blok {a.block} byte ==")
    print(f"{'byte':>9}{'BPB blok':>11}{'BPB kumulatif':>16}" + ("   ng_lam" if lams else ""))
    bounds = list(range(0, total_bytes, a.block)) + [total_bytes]
    for k in range(len(bounds) - 1):
        b0, b1 = bounds[k], bounds[k + 1]
        i0 = int(np.searchsorted(cbytes, b0, side="right"))
        i1 = int(np.searchsorted(cbytes, b1, side="right"))
        seg = bits[i0:i1]
        ok = ~np.isnan(seg)
        if not ok.any():
            continue
        blk = float(seg[ok].sum()) / float(blens[i0:i1][ok].sum())
        cum = float(np.nansum(bits[:i1])) / float(blens[:i1].sum())
        tail = f"   {lams[min(i1, len(lams)) - 1]:.4f}" if lams else ""
        print(f"{b1:>9}{blk:>11.3f}{cum:>16.3f}{tail}")

    bm, bl = bits[1:], blens[1:]
    print(f"\n== BPB prekuensial (semakin kecil semakin baik) ==")
    print(f"{'model':<22}{'semua':>9}{'sepertiga akhir':>18}")
    i_third = int(np.searchsorted(cbytes, total_bytes // 3, side="right"))
    rows = [("TMT2 (FastLearner)", bm, bl)]
    for o in (0, 1, 2):
        nb = ngram_bits(canon, o)[1:]
        rows.append((f"n-gram order {o}", nb, np.ones(len(nb))))
    for name, arr, lens in rows:
        ok = ~np.isnan(arr)
        tot = float(arr[ok].sum()) / float(lens[ok].sum())
        seg, sl = arr[i_third:], lens[i_third:]
        ok2 = ~np.isnan(seg)
        print(f"{name:<22}{tot:>9.3f}{float(seg[ok2].sum())/float(sl[ok2].sum()):>18.3f}")

    print(f"\n== contoh generasi ({a.gen_len} token, think=False) ==")
    prompt = [int(x) for x in data[:32]]
    for k, (sd, temp) in enumerate([(1, 0.7), (2, 0.9), (3, 0.5)][:a.gen]):
        g = ln.generate(prompt, a.gen_len, temp=temp, think=False, seed=sd)
        txt = tk.decode(tok, g) if a.bpe else repr(bytes(g))
        print(f"[{k+1}] temp={temp} seed={sd}: {txt!r}")

    if a.out:
        ln.save(a.out)
        print(f"\ncheckpoint -> {a.out}")


if __name__ == "__main__":
    main()
