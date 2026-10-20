"""Probe oracle: memori dipasang ANALITIK, hanya jalur baca/keluaran yang dilatih.

Memisahkan "dapat dinyatakan" dari "dapat dipelajari". `E` dibuat ortogonal (interferensi
memori nol) dan `d_mem = d` sehingga proyeksi identitas. Yang ditetapkan/dilatih diatur
oleh `--learn`:

  none  Wk=Wq=Wv=I         (oracle penuh; Wq TIDAK dilatih -> identitas paksa)
  q     Wk=Wv=I, Wq acak   -> Wq DILATIH. Ini perbandingan x_t vs h_t yang ADIL:
                               dengan `none`, identitas atas `ln(h_t)` adalah proyeksi
                               sembarang atas campuran berpeluruhan, jadi gagal hampir pasti
                               apa pun yang bisa dipelajari model -- hasil "h_t datar" dari
                               `none` TIDAK membuktikan h tak bisa dibaca Wq terlatih.
  qkv   Wk/Wv/Wq acak, dilatih (gradien Wk/Wv terpangkas 1 langkah)
  all   + E dilatih (memori dari nol)

  python scripts/oracle_probe.py --Ls 4,8 --seeds 3 --learn none --beta_max 1.0

LANTAI: pada paruh 2, posisi pertama blok kedua (blk[0]) diprediksi dari blk[L-1]; penerus
blk[0] belum pernah terlihat -> tidak bisa dipanggil dari memori. Biaya ~log2(V) untuk 1 dari
tiap L posisi => lantai ~ log2(V)/L. "Ideal 0 bit" SALAH; lihat kolom `lantai`.
Untuk V=4 non-perm lantainya lebih rumit (ambiguitas kunci berulang).
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tmt2.model import Config, TMT2
from tmt2.learner import Stream, TrainOpts
from tmt2.fast_learner import FastLearner
from tmt2.corpus import copy_task


def run(L, seed, n, V, d, lr, slow_every, query_x, beta_max=0.5, learn="none", perm=True,
        wk_init="I", wq_init="rand"):
    ids, second = copy_task(L, n, V, seed, perm=perm)
    cfg = Config(d=d, d_mem=d, n_layers=3, vocab=V, head="linear", lr=lr, warmup=20,
                 slow_every=slow_every, replay=False, seed=seed, deep_grad=True,
                 mem_hetero=True, mem_surprise_fixed=True, mem_query_x=query_x,
                 beta_max=beta_max, use_innovation=False, adapter_rank=0,
                 mem_write_grad=(learn in ("qkv", "all")))
    m = TMT2(cfg)
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    m.E.data[:] = Q[:V] * np.sqrt(d)                      # baris ortogonal (butuh V <= d)
    Rk = rng.normal(0.0, 1.0 / np.sqrt(d), (d, d))        # kandidat Wk acak
    Ro = rng.normal(0.0, 1.0 / np.sqrt(d), (d, d))        # kandidat Wq acak
    for Lyr in m.layers:
        if learn == "none":
            Lyr.Wk[:] = np.eye(d)
            Lyr.Wv[:] = np.eye(d)
            Lyr.Wq.data[:] = np.eye(d)
            continue
        if learn == "q":                                  # Wk=Wv dibekukan, hanya Wq dilatih
            Lyr.Wk[:] = np.eye(d) if wk_init == "I" else Rk
            Lyr.Wv[:] = np.eye(d)
        # "qkv"/"all": Wk/Wv/Wq acak (Wk/Wv jadi Param); yang membedakan hanya daftar `only`
        if wq_init == "I":
            Lyr.Wq.data[:] = np.eye(d)
        elif wq_init == "ortho":
            Lyr.Wq.data[:] = Q
        elif wq_init == "match_k":                        # SELARAS dengan ruang tulis
            Lyr.Wq.data[:] = getattr(Lyr.Wk, "data", Lyr.Wk)
        else:
            Lyr.Wq.data[:] = Ro
    top = cfg.n_layers - 1
    only = ["W_out", "byte_bias", f"P{top}_1"] + [
        f"L{l}.{k}" for l in range(cfg.n_layers) for k in ("Wc", "Wm", "Wg1", "Wg2", "Wo")]
    if learn in ("q", "qkv", "all"):
        only += [f"L{l}.Wq" for l in range(cfg.n_layers)]
    if learn in ("qkv", "all"):
        only += [f"L{l}.{k}" for l in range(cfg.n_layers) for k in ("Wk", "Wv")]
    if learn == "all":
        only += ["E"]
    ln = FastLearner(m, TrainOpts(seed=seed, think_prob=0.0, use_stop=False,
                                  only_byte_loss=True, only_params=tuple(only)))
    s, bits = Stream(m), np.full(n, np.nan)
    for i, b in enumerate(ids):
        c = ln.step(s, int(b))
        if c is not None:
            bits[i] = c / np.log(2)
    h = n // 2
    ok = ~np.isnan(bits)
    return float(bits[h:][second[h:] & ok[h:]].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ls", default="4,8")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--V", type=int, default=32)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--slow_every", type=int, default=4)
    ap.add_argument("--beta_max", type=float, default=0.5)
    ap.add_argument("--learn", default="none", choices=["none", "q", "qkv", "all"])
    ap.add_argument("--wk_init", default="I", choices=["I", "rand"],
                    help="init Wk (beku kecuali learn=qkv/all)")
    ap.add_argument("--wq_init", default="rand", choices=["rand", "I", "ortho", "match_k"],
                    help="init Wq saat learn=q. match_k = salin Wk -> baca/tulis selaras")
    a = ap.parse_args()
    if a.V > a.d:
        ap.error("butuh V <= d untuk E ortogonal")
    lantai = np.log2(a.V)
    print(f"lantai = log2(V)/L ; tanpa memori = {lantai:.2f} bit (seragam). "
          f"beta_max={a.beta_max} learn={a.learn} wk={a.wk_init} wq={a.wq_init} d={a.d} "
          f"n={a.n} lr={a.lr} slow_every={a.slow_every}")
    print(f"{'L':>4}{'lantai':>8}{'kueri':>8}{'paruh2 (mean±sd)':>22}{'ke lantai':>11}")
    for L in [int(x) for x in a.Ls.split(",")]:
        fl = lantai / L
        for qx in (True, False):
            r = np.array([run(L, sd, a.n, a.V, a.d, a.lr, a.slow_every, qx,
                              a.beta_max, a.learn, True, a.wk_init, a.wq_init)
                          for sd in range(a.seeds)])
            print(f"{L:>4}{fl:>8.2f}{'x_t' if qx else 'h_t':>8}"
                  f"{r.mean():>14.3f} ± {r.std():.3f}{r.mean()-fl:>11.3f}", flush=True)


if __name__ == "__main__":
    main()

