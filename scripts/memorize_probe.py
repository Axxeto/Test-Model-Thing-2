"""UJI HAFALAN BERSIH: bisakah TMT2 menghafal 512 byte ACAK V=64 sampai <0.1 bit?

Beda dari `overfit_probe.py` / `overfit_unit.py`:
  * `eval_stream` WAJIB konsisten dengan mode latih (think_prob=0 -> K=0). Sejak
    perbaikan mode-berpikir di `FastLearner.step`, default sudah benar. Skrip ini
    juga mencetak `fz_think` (think=True dipaksa) sebagai kontrol diagnostik.
  * Melacak kurva CE BEKU tiap `eval_every` epoch untuk melihat APAKAH dan KAPAN
    plateau terjadi, bukan hanya nilai akhir.
  * Mencetak norma gradien lapisan bawah & atas untuk memisahkan "gradien hilang"
    dari "kapasitas/optimisasi".

Kriteria (disepakati): HAFAL bila CE beku < 0.1 bit dalam <= 200 epoch.
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tmt2.model import Config, TMT2
from tmt2.learner import Stream, TrainOpts
from tmt2.fast_learner import FastLearner
from tmt2.corpus import random_bytes

LN2 = float(np.log(2))


def grad_norms(model):
    """Norma gradien (kalau ada) untuk kepala vs lapisan bawah.

    CATATAN: `Param` TIDAK punya atribut `.g`; gradien terakumulasi di dict `p.tg`
    (kunci = nama task: "byte"/"jepa"/...). Versi lama skrip ini membaca `p.g` ->
    AttributeError pada eval verbose pertama, sehingga skrip tidak pernah selesai.
    """
    out = {}
    for name in ("W_out", "E", "Wa", "Wc"):
        p = model.params.get(name)
        if p is None or not getattr(p, "tg", None):
            continue
        gsum = None
        for g in p.tg.values():
            gsum = g if gsum is None else gsum + g
        out[name] = float(np.linalg.norm(gsum))
    return out


def run(data, L, dp, se, epochs, d=32, lr=3e-2, d_mem=8, eval_every=20, seed=0, verbose=True):
    cfg = Config(d=d, n_layers=L, d_mem=d_mem, head="linear", deep_grad=bool(dp), lr=lr,
                 warmup=200, slow_every=se, clip=1.0, wd=0.0, replay=False, adapter_rank=0,
                 use_innovation=False, mem_query_x=True, mem_wq_match_k=True, mem_hetero=True,
                 mem_surprise_fixed=True, beta_max=1.0, seed=seed)
    m = TMT2(cfg)
    ln = FastLearner(m, TrainOpts(seed=seed, think_prob=0.0, use_stop=False, only_byte_loss=True))
    s = Stream(m)
    target = [int(b) for b in data]
    t0 = time.time()
    train_curve, frozen_curve = [], []
    gnorm = None
    for ep in range(epochs):
        ce = [c for c in (ln.step(s, int(b)) for b in data) if c is not None]
        train_curve.append(float(np.mean(ce) / LN2))
        if (ep + 1) % eval_every == 0 or ep == epochs - 1:
            fz = float(ln.eval_stream(target).mean() / LN2)
            frozen_curve.append((ep + 1, fz))
            if verbose:
                gn = grad_norms(m)
                gs = " ".join(f"{k}={v:.2e}" for k, v in gn.items())
                print(f"    ep{ep+1:>3}  latih={train_curve[-1]:.4f}  beku={fz:.4f}   {gs}", flush=True)
            if gnorm is None:
                gnorm = grad_norms(m)
    fz_final = frozen_curve[-1][1] if frozen_curve else float("nan")
    # kontrol diagnostik: berapa bila think dipaksa (seharusnya JAUH lebih buruk)
    fz_think = float(ln.eval_stream(target, think=True).mean() / LN2)
    return dict(frozen=fz_final, train_last=train_curve[-1], fz_think=fz_think,
                frozen_curve=frozen_curve, train_curve=train_curve,
                secs=time.time() - t0, gnorm=gnorm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=3e-2)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--vocab", type=int, default=64)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--d_mem", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--grid", default="1:0:16,1:1:16,3:0:16,3:1:16",
                    help="L:deep:slow_every, dipisah koma")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    data = random_bytes(a.n, a.vocab, a.seed)
    print(f"data: {a.n} byte ACAK V={a.vocab} (entropi={np.log2(a.vocab):.1f} bit/byte) "
          f"- HANYA bisa diturunkan dengan menghafal")
    print(f"kriteria HAFAL: CE beku < 0.1 bit dalam <= {a.epochs} epoch\n")
    hdr = f"{'L':>2}{'deep':>5}{'se':>4}{'beku':>9}{'latih':>9}{'beku_think':>11}{'detik':>7}"
    print(hdr); print("-" * len(hdr))
    results = []
    for spec in a.grid.split(","):
        L, dp, se = (int(x) for x in spec.split(":"))
        print(f"  [L={L} deep={dp} se={se}]", flush=True)
        r = run(data, L, dp, se, a.epochs, d=a.d, lr=a.lr, d_mem=a.d_mem)
        results.append((spec, r))
        print(f"{L:>2}{dp:>5}{se:>4}{r['frozen']:>9.4f}{r['train_last']:>9.4f}"
              f"{r['fz_think']:>11.4f}{r['secs']:>7.0f}", flush=True)
        fz = r["frozen"]
        verdict = "HAFAL" if fz < 0.1 else ("mendekati" if fz < 0.5 else "GAGAL menghafal")
        print(f"    -> {verdict}  (latih={r['train_last']:.4f}, selisih latih-beku={r['train_last']-fz:+.4f})\n",
              flush=True)

    print("=" * 64)
    print("RINGKASAN")
    for spec, r in results:
        fc = r["frozen_curve"]
        traj = " ".join(f"{e}:{v:.2f}" for e, v in fc)
        print(f"  L/deep/se={spec:<8} beku {r['frozen']:.4f} | kurva beku {traj}")
    best = min(results, key=lambda kv: kv[1]["frozen"])
    print(f"\nTERBAIK: {best[0]} -> CE beku {best[1]['frozen']:.4f} bit")


if __name__ == "__main__":
    main()
