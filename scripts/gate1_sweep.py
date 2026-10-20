"""GATE 1 — baseline adil, ≥3 seed × ≥2 korpus, dijalankan PARALEL.

Protokol (sama untuk semua): "latih" pada data[:ntrain], nilai pada data[ntrain:ntrain+ntest]
yang TAK PERNAH dilihat. Baseline n-gram/ctx-mix dibekukan setelah fase latih; kompresor diberi
protokol KONDISIONAL (bit tambahan untuk mengkode tr+te setelah tr dikode); BPE dilatih HANYA
pada tr.

Paralelisme = level EKSPERIMEN (multiprocessing): inti model sendiri tetap single-thread.

  python scripts/gate1_sweep.py --seeds 0,1,2 --ntrain 200000 --ntest 8000 --procs 7
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from multiprocessing import Pool

from tmt2.model import Config, TMT2
from tmt2.corpus import load
from tmt2.learner import Stream, TrainOpts
from tmt2.fast_learner import FastLearner

LN2 = float(np.log(2))
CORPORA = {"wikitext": "wikitext", "indonesian": "data/indonesian_1m_pure.txt"}


# ---------------------------------------------------------------- pekerja TMT2
def job_tmt2(spec):
    corpus, seed, ntrain, ntest, d, nlayers = spec
    raw = load(CORPORA[corpus])
    tr = raw[:ntrain]
    te = raw[ntrain:ntrain + ntest]
    cfg = Config.best(d=d, n_layers=nlayers, vocab=256, d_mem=max(8, d // 4), use_memory=True,
                      head="linear", deep_grad=True, adapter_rank=0, think_max=0,
                      use_innovation=False, seg_len=10 ** 9, seed=seed)
    m = TMT2(cfg)
    ln = FastLearner(m, TrainOpts(only_byte_loss=True, seed=seed, use_stop=False))
    s = Stream(m, online=True)
    for b in tr:
        ln.step(s, int(b), train=True)
    ces = []
    for b in te:
        c = ln.step(s, int(b), train=False)
        if c is not None:
            ces.append(c)
    return ("tmt2", corpus, seed, float(np.mean(ces) / LN2))


# ---------------------------------------------------------------- baseline byte
def job_ngram(spec):
    corpus, ntrain, ntest = spec
    from fair_online_compare import jm_ngram
    raw = np.frombuffer(load(CORPORA[corpus]), dtype=np.uint8)
    tr, te = raw[:ntrain], raw[ntrain:ntrain + ntest]
    return ("ngram_jm5", corpus, -1, float(jm_ngram(tr, te)))


def job_ctxmix(spec):
    corpus, ntrain, ntest = spec
    from fair_online_compare import ctxmix_frozen
    raw = np.frombuffer(load(CORPORA[corpus]), dtype=np.uint8)
    tr, te = raw[:ntrain], raw[ntrain:ntrain + ntest]
    return ("ctxmix4", corpus, -1, float(ctxmix_frozen(tr, te)))


def job_comp(spec):
    corpus, ntrain, ntest = spec
    import bz2, lzma, zlib
    raw = load(CORPORA[corpus])
    tr, te = bytes(raw[:ntrain]), bytes(raw[ntrain:ntrain + ntest])
    out = []
    for name, fn in (("bz2", lambda b: bz2.compress(b, 9)),
                     ("lzma", lambda b: lzma.compress(b, preset=9)),
                     ("zlib", lambda b: zlib.compress(b, 9))):
        v = 8.0 * (len(fn(tr + te)) - len(fn(tr))) / len(te)
        out.append((f"{name}_kondisional", corpus, -1, float(v)))
    return out


def job_bpe(spec):
    corpus, ntrain, ntest, vocab = spec
    from fair_bpe_split import bits_frozen
    from tmt2 import tokenizer as tk
    raw = load(CORPORA[corpus])
    tr = bytes(raw[:ntrain])
    te = bytes(raw[ntrain:ntrain + ntest])
    tok = tk.train(tr, vocab, None)
    ids_tr, _, _ = tk.encode(tok, tr)
    ids_te, lens_te, _ = tk.encode(tok, te)
    ids_tr = [int(x) for x in ids_tr]
    ids_te = [int(x) for x in ids_te]
    b_te = int(np.sum(lens_te))
    best = None
    for o in (1, 2):
        bits = bits_frozen(ids_tr, ids_te, o, vocab)
        v = float(bits.sum() / b_te)
        if best is None or v < best:
            best = v
    return (f"bpe_token_best", corpus, -1, best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--ntrain", type=int, default=200000)
    ap.add_argument("--ntest", type=int, default=8000)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--bpe_vocab", type=int, default=4096)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--corpora", default="wikitext,indonesian")
    a = ap.parse_args()

    seeds = [int(x) for x in a.seeds.split(",")]
    corpora = [c.strip() for c in a.corpora.split(",") if c.strip()]
    print(f"GATE 1 — ntrain={a.ntrain} ntest={a.ntest} d={a.d} L={a.layers} "
          f"seeds={seeds} procs={a.procs}", flush=True)

    tmt2_jobs = [(c, sd, a.ntrain, a.ntest, a.d, a.layers) for c in corpora for sd in seeds]
    base_jobs = [(c, a.ntrain, a.ntest) for c in corpora]
    bpe_jobs = [(c, a.ntrain, a.ntest, a.bpe_vocab) for c in corpora]

    t0 = time.time()
    with Pool(a.procs) as pool:
        tmt2_res = pool.map(job_tmt2, tmt2_jobs)
        base_res = pool.map(job_ngram, base_jobs) + pool.map(job_ctxmix, base_jobs)
        for lst in pool.map(job_comp, base_jobs):
            base_res += lst
        bpe_res = pool.map(job_bpe, bpe_jobs)
    print(f"(total {time.time()-t0:.0f} detik, {a.procs} proses)\n", flush=True)

    results = {}
    for name, corpus, seed, v in tmt2_res + base_res + bpe_res:
        results.setdefault((corpus, name), []).append(v)

    base_names = ["ngram_jm5", "ctxmix4", "bz2_kondisional", "lzma_kondisional",
                  "zlib_kondisional", "bpe_token_best"]
    all_pass = True
    for c in corpora:
        tvals = np.array(results[(c, "tmt2")])
        print(f"=== korpus: {c} ===")
        print(f"  {'TMT2+mixer':<24}{tvals.mean():>9.4f} ± {tvals.std():.4f}  "
              f"(seed: {' / '.join(f'{v:.4f}' for v in tvals)})")
        for nm in base_names:
            if (c, nm) not in results:
                continue
            v = float(np.mean(results[(c, nm)]))
            win = tvals.mean() < v
            all_pass &= win
            print(f"  {nm:<24}{v:>9.4f}   {'MENANG' if win else 'KALAH'} "
                  f"(margin {tvals.mean()-v:+.4f})")
        print(flush=True)

    print("== PUTUSAN GATE 1 ==")
    print(f"  TMT2+mixer menang di SEMUA baseline, SEMUA korpus, ≥{len(seeds)} seed: "
          f"{'LULUS' if all_pass else 'BELUM'}")


if __name__ == "__main__":
    main()
