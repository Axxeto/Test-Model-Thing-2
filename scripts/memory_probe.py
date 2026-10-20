"""Probe memori: tugas salin dengan lag L x sapuan s_decay.

Blok acak panjang L diulang sekali; model hanya bisa menang dengan menyimpan blok itu di state
lalu membacanya kembali setelah lag L. Batas ideal: paruh kedua 0 bit, paruh pertama log2(V).

  python scripts/memory_probe.py --Ls 8,24,64 --decays 0,0.5,0.9,0.99 --seeds 2

Cara membaca:
  * titik patah (L terbesar yang masih berhasil) bergeser ke L lebih besar saat s_decay naik
    -> kredit gradien terpotong; s_decay hambatan nyata.
  * titik patah sama di semua s_decay -> kapasitas/mekanisme memori yang membatasi.
  * paruh2 ~ paruh1 bahkan pada L=8 -> model tidak bisa membaca konteks sama sekali.
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tmt2.model import Config, TMT2
from tmt2.learner import Stream, TrainOpts
from tmt2.fast_learner import FastLearner
from tmt2.corpus import copy_task


def run(L, sd, seed, n, V, d, slow_every, lr, deep_grad=True, mem_write_grad=False,
        mem_hetero=False, mem_surprise_fixed=False, perm=False, cycle=False, trace=False,
        mem_query_x=False, mem_wq_match_k=False, beta_max=0.5):
    ids, second = copy_task(L, n, V, seed, perm=perm, cycle=cycle)
    cfg = Config(d=d, n_layers=3, d_mem=max(8, d // 4), vocab=V, head="linear",
                 lr=lr, warmup=20, slow_every=slow_every, s_decay=sd, replay=False, seed=seed,
                 deep_grad=deep_grad, mem_write_grad=mem_write_grad, mem_hetero=mem_hetero,
                 mem_surprise_fixed=mem_surprise_fixed, mem_query_x=mem_query_x,
                 mem_wq_match_k=mem_wq_match_k, beta_max=beta_max)
    m = TMT2(cfg)
    ln = FastLearner(m, TrainOpts(seed=seed, think_prob=0.0, use_stop=False, only_byte_loss=True))
    s, bits = Stream(m), np.full(n, np.nan)
    for i, b in enumerate(ids):
        c = ln.step(s, int(b))
        if c is not None:
            bits[i] = c / np.log(2)
    half = n // 2                                   # ukur hanya separuh akhir aliran
    ok = ~np.isnan(bits)
    b2 = bits[half:][second[half:] & ok[half:]].mean()
    b1 = bits[half:][~second[half:] & ok[half:]].mean()
    if not trace:
        return float(b1), float(b2)
    # jejak: bit kumulatif paruh2 vs jumlah token yang DIPREDIKSI (bukan posisi byte),
    # supaya waktu-konvergensi bisa dibaca sebagai "butuh berapa token untuk turun ke X bit".
    pos, cum, tot = [], [], 0.0
    for i in range(half, n):
        if ok[i] and second[i]:
            tot += bits[i]
            pos.append(i - half)
            cum.append(tot / len(pos))
    return (float(b1), float(b2), np.asarray(pos), np.asarray(cum))


def _first_below(pos, cum, thr):
    """Indeks dalam `pos` saat rata-rata kumulatif pertama kali < thr, atau None."""
    hit = np.nonzero(cum < thr)[0]
    return int(pos[hit[0]]) if hit.size else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ls", default="8,24,64")
    ap.add_argument("--decays", default="0,0.5,0.9,0.99")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--V", type=int, default=4)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--slow_every", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--no_deep_grad", action="store_true",
                    help="matikan deep_grad (default: AKTIF, karena tanpa itu layer bawah tak terlatih)")
    ap.add_argument("--mem_write_grad", action="store_true",
                    help="Wk/Wv jadi Param: penulisan memori dapat dipelajari")
    ap.add_argument("--mem_hetero", action="store_true",
                    help="kunci tulis dari konteks SEBELUMNYA (asosiasi hetero)")
    ap.add_argument("--mem_surprise_fixed", action="store_true",
                    help="beta = beta_max tetap (lepas dari gerbang kejutan)")
    ap.add_argument("--perm", action="store_true",
                    help="kunci 1-token UNIK di dalam blok (butuh L<=V); menghapus ambiguitas "
                         "'hafalan vs prediksi 1 langkah' pada V kecil")
    ap.add_argument("--cycle", action="store_true",
                    help="dengan --perm: pakai SATU permutasi untuk semua blok (kontrol negatif)")
    ap.add_argument("--mem_query_x", action="store_true",
                    help="kueri baca dari ln(x_t) (token mentah), bukan ln(h_t)")
    ap.add_argument("--mem_wq_match_k", action="store_true",
                    help="init Wq = Wk (selaraskan ruang baca & tulis). Terbukti di oracle: "
                         "Wq acak -> 5,01; Wq=Wk -> 2,55")
    ap.add_argument("--beta_max", type=float, default=0.5)
    ap.add_argument("--tag", default="",
                    help="label yang dicetak di awal tiap baris (untuk menyatukan tabel manual)")
    ap.add_argument("--trace", action="store_true",
                    help="cetak jumlah token sampai paruh2 kumulatif pertama kali < ambang")
    ap.add_argument("--trace_thr", default="1.0,2.0,3.0",
                    help="ambang bit untuk --trace (daftar dipisah koma)")
    a = ap.parse_args()
    if a.perm and max(int(x) for x in a.Ls.split(",")) > a.V:
        ap.error(f"--perm butuh semua L <= V (V={a.V})")
    print(f"ideal: paruh1={np.log2(a.V):.2f} bit, paruh2=0.00 bit   d={a.d} "
          f"slow_every={a.slow_every} V={a.V} n={a.n} deep_grad={not a.no_deep_grad} "
          f"hetero={a.mem_hetero} beta_fix={a.mem_surprise_fixed} perm={a.perm} "
          f"cycle={a.cycle}")
    print(f"{'L':>4}{'s_decay':>9}{'paruh1':>9}{'paruh2 (mean±sd)':>22}{'detik':>8}"
          + (f"   {a.tag}" if a.tag else ""))
    for L in [int(x) for x in a.Ls.split(",")]:
        for sd in [float(x) for x in a.decays.split(",")]:
            t0 = time.time()
            out = [run(L, sd, s, a.n, a.V, a.d, a.slow_every, a.lr,
                       not a.no_deep_grad, a.mem_write_grad,
                       a.mem_hetero, a.mem_surprise_fixed, a.perm, a.cycle, a.trace,
                       a.mem_query_x, a.mem_wq_match_k, a.beta_max)
                   for s in range(a.seeds)]
            if a.trace:
                hr = [x[1] for x in out]
                et = time.time() - t0
                print(f"{L:>4}{sd:>9.2f}{np.mean([x[0] for x in out]):>9.3f}"
                      f"{np.mean(hr):>14.3f} ± {np.std(hr):.3f}{et:>8.1f}"
                      + (f"   {a.tag}" if a.tag else ""), flush=True)
                thrs = [float(x) for x in a.trace_thr.split(",")]
                for thr in thrs:
                    firsts = [_first_below(x[2], x[3], thr) for x in out]
                    ok = [f for f in firsts if f is not None]
                    if ok:
                        print(f"       paruh2 < {thr:.2f} bit setelah "
                              f"{np.mean(ok):>7.0f} token (mean {len(ok)}/{len(firsts)} seed)")
                    else:
                        print(f"       paruh2 < {thr:.2f} bit: TIDAK PERNAH "
                              f"(akhir {np.mean(hr):.3f})")
            else:
                r = np.array([(x[0], x[1]) for x in out])
                print(f"{L:>4}{sd:>9.2f}{r[:, 0].mean():>9.3f}{r[:, 1].mean():>14.3f} "
                      f"± {r[:, 1].std():.3f}{time.time()-t0:>8.1f}"
                      + (f"   {a.tag}" if a.tag else ""), flush=True)


if __name__ == "__main__":
    main()
