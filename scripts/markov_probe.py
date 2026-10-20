"""PROBE MARKOV ORDE-k — sumber dengan lantai entropi ANALITIK dan tanpa sidik jari posisi.

Mengapa ini menggantikan probe "byte acak":
  * Byte acak i.i.d. hanya menguji HAFALAN, bukan pemakaian konteks.
  * Aliran acak berulang (copy_task) bisa dihafal lewat SIDIK JARI POSISI di state lambat,
    tanpa resolusi konteks sama sekali. Itu tidak membedakan "konteks" dari "posisi".
  * Sumber Markov: simbol berikutnya bergantung pada k byte terakhir, dengan peluang noise eps.
    - Lantai entropi DIKETAHUI analitik (bisa dibandingkan langsung dengan CE model).
    - Aliran non-periodik -> tidak ada sidik jari posisi.
    - Kurva "orde konteks efektif" bisa dibaca langsung dari k=1..4.

Model: V=8 (3 bit/simbol). Tabel acak T[konteks k] -> simbol; dengan peluang (1-eps) keluarkan
T[konteks], dengan peluang eps keluarkan simbol acak seragam. Konteks = k simbol terakhir.

Lantai entropi per simbol (kondisional):
    H = (1-eps)*0 + eps*log2(V)      ... bila T deterministik dan konteks teramati
Jadi target CE model ~= eps*log2(V) = eps*3 bit. Untuk eps=0.1 -> 0.3 bit.
Model yang hanya memakai konteks orde j < k gagal menutup sisa entropi -> CE lebih tinggi.
Dengan membandingkan CE ke lantai pada k=1,2,3,4 kita baca "orde konteks efektif".

Protokol: latih satu lintasan (online), lalu ukur CE beku pada DUA mode (dingin & hangat).
Catat juga CE latih. Semua hasil -> JSONL.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tmt2.model import Config, TMT2
from tmt2.learner import Stream, TrainOpts
from tmt2.fast_learner import FastLearner
from tmt2.corpus import sanity_check

LN2 = float(np.log(2))


def markov_stream(n, k, V=8, eps=0.1, seed=0, n_eval=0, eval_seed=None):
    """Bangkitkan aliran Markov orde-k.

    ``n_eval`` > 0: kembalikan ``(train_ids, eval_ids, floor)`` di mana ``eval_ids`` adalah
    LANJUTAN SEGAR dari TABEL YANG SAMA (bukan ulangan data latih). Ini penting: mengevaluasi
    pada data latih yang diulang bisa dihafal lewat sidik jari posisi -- persis yang mau dihindari.

    ``n_eval`` == 0: perilaku lama, kembalikan ``(ids, floor)``.
    """
    rng = np.random.default_rng(seed)
    nCtx = V ** k
    table = rng.integers(0, V, nCtx)                       # T[konteks] -> simbol

    def gen(count, ctx, rng_):
        out = np.empty(count, dtype=np.int64)
        for i in range(count):
            s = int(rng_.integers(0, V)) if rng_.random() < eps else int(table[ctx])
            out[i] = s
            ctx = (ctx * V + s) % nCtx
        return out, ctx

    train_ids, ctx = gen(n, 0, rng)
    floor = eps * np.log2(V)
    if n_eval <= 0:
        return train_ids, float(floor)
    erng = np.random.default_rng(eval_seed if eval_seed is not None else seed + 10_000)
    eval_ids, _ = gen(n_eval, ctx, erng)                   # lanjut dari konteks terakhir
    return train_ids, eval_ids, float(floor)


def analytic_floor_uniform(V):
    return float(np.log2(V))


def run(n, k, V, eps, epochs, d, L, d_mem, se, deep, lr, warm, seed, tag="", n_eval=0):
    train_ids, eval_ids, floor = markov_stream(n, k, V, eps, seed, n_eval=n_eval)
    sanity_check(np.frombuffer(np.asarray(train_ids, dtype=np.uint8).tobytes(), np.uint8),
                 vocab=V, name=f"markov k={k}", expect_len=n)
    if n_eval <= 0:
        # perilaku lama (data latih dipakai ulang untuk eval) -- BERI TANDA agar tidak disalahartikan
        eval_ids = train_ids
    cfg = Config(d=d, n_layers=L, d_mem=d_mem, head="linear", deep_grad=bool(deep), lr=lr,
                 warmup=200, slow_every=se, clip=1.0, wd=0.0, replay=False, adapter_rank=0,
                 use_innovation=False, mem_query_x=True, mem_wq_match_k=True, mem_hetero=True,
                 mem_surprise_fixed=True, beta_max=1.0, seed=seed)
    m = TMT2(cfg)
    ln = FastLearner(m, TrainOpts(seed=seed, think_prob=0.0, use_stop=False, only_byte_loss=True))
    s = Stream(m)
    t0 = time.time()
    train_ce = None
    for _ in range(epochs):
        ce = [c for c in (ln.step(s, int(b)) for b in train_ids) if c is not None]
        train_ce = float(np.mean(ce) / LN2)
    # eval pada aliran SEGAR dari tabel yang sama (bukan data latih)
    cold = float(ln.eval_stream([int(b) for b in eval_ids]).mean() / LN2)
    sw = Stream(m, online=False)
    for b in eval_ids[:min(warm, len(eval_ids) - 1)]:
        ln.step(sw, int(b), train=False, think=False)
    out = []
    for b in eval_ids[min(warm, len(eval_ids) - 1):]:
        c = ln.step(sw, int(b), train=False, think=False)
        if c is not None:
            out.append(c)
    hot = float(np.mean(out) / LN2)
    return dict(n=n, k=k, V=V, eps=eps, epochs=epochs, d=d, L=L, d_mem=d_mem, se=se,
                deep=int(deep), lr=lr, warm=warm, seed=seed, tag=tag,
                floor=floor, uniform=analytic_floor_uniform(V), n_eval=n_eval,
                eval_on_train=(n_eval <= 0),
                train_ce=train_ce, cold=cold, hot=hot, secs=time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--ks", default="1,2,3,4", help="orde konteks Markov yang diuji")
    ap.add_argument("--V", type=int, default=8)
    ap.add_argument("--eps", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--L", type=int, default=3)
    ap.add_argument("--d_mem", type=int, default=8)
    ap.add_argument("--se", type=int, default=16)
    ap.add_argument("--deep", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--warm", type=int, default=256)
    ap.add_argument("--n_eval", type=int, default=4096,
                    help="byte uji dari aliran SEGAR (0 = pakai data latih, TIDAK disarankan)")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--jsonl", default="results/markov_probe.jsonl")
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.jsonl) or ".", exist_ok=True)
    seeds = [int(x) for x in a.seeds.split(",")]
    ks = [int(x) for x in a.ks.split(",")]
    print(f"Markov V={a.V} eps={a.eps} n={a.n} epochs={a.epochs} seeds={seeds}")
    print(f"lantai analitik = eps*log2(V) = {a.eps*np.log2(a.V):.4f} bit/simbol\n")
    hdr = f"{'k':>2}{'seed':>5}{'lantai':>8}{'latih':>8}{'beku_dingin':>12}{'beku_hangat':>12}{'detik':>7}"
    print(hdr); print("-" * len(hdr))
    rows = []
    with open(a.jsonl, "a") as fh:
        for k in ks:
            for sd in seeds:
                r = run(a.n, k, a.V, a.eps, a.epochs, a.d, a.L, a.d_mem, a.se, a.deep,
                        a.lr, a.warm, sd, tag="markov", n_eval=a.n_eval)
                rows.append(r)
                fh.write(json.dumps(r) + "\n")
                print(f"{k:>2}{sd:>5}{r['floor']:>8.4f}{r['train_ce']:>8.4f}"
                      f"{r['cold']:>12.4f}{r['hot']:>12.4f}{r['secs']:>7.0f}", flush=True)
    print("\nRINGKASAN (rata-rata lintas seed, eval hangat):")
    for k in ks:
        hots = [r["hot"] for r in rows if r["k"] == k]
        print(f"  k={k}: hangat={np.mean(hots):.4f} bit (lantai {rows[0]['floor']:.4f})")
    print("\nTafsir: bila CE hangat(k) turun mendekati lantai hanya sampai k=K0 lalu datar,")
    print("orde konteks efektif model ~ K0. Eval hangat vs dingin memisahkan efek A2.")


if __name__ == "__main__":
    main()
