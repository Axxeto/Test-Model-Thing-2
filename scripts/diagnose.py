"""Eksperimen uji hipotesis kepala keluaran TMT2 (Tugas 4).

  python scripts/diagnose.py --label baseline
  python scripts/diagnose.py --label A_pcgrad_off --pcgrad_off
  python scripts/diagnose.py --label B_Wout --separate_out --pcgrad_off
  python scripts/diagnose.py --label C_outlr10 --out_lr 10 --pcgrad_off --debug_out

Metrik: BPB (semua / sepertiga akhir), akurasi argmax PREKUENSIAL (prediksi diukur sebelum
byte-nya terlihat), jumlah byte unik yang diprediksi, entropi distribusi prediksi,
logit_scale, dan CE(softmax(byte_bias)) sebagai cek kalibrasi bias keluaran.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tmt2.model import Config, TMT2
from tmt2.learner import Stream, TrainOpts
from tmt2.fast_learner import FastLearner
from tmt2.corpus import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run")
    ap.add_argument("--n", type=int, default=46507)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--slow_every", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="license")
    ap.add_argument("--pcgrad_off", action="store_true")
    ap.add_argument("--separate_out", action="store_true")
    ap.add_argument("--head", default="cosine", choices=["cosine", "linear"])
    ap.add_argument("--out_lr", type=float, default=1.0)
    ap.add_argument("--init_scale", type=float, default=3.0, help="logit_scale awal (exp)")
    ap.add_argument("--debug_out", action="store_true")
    ap.add_argument("--ckpt", default=None)
    a = ap.parse_args()

    data = load(a.data)[:a.n]
    cfg = Config(d=a.d, n_layers=3, d_mem=max(8, a.d // 4), lr=3e-2, warmup=20,
                 slow_every=a.slow_every, seed=a.seed,
                 pcgrad=not a.pcgrad_off, separate_out=a.separate_out, head=a.head)
    model = TMT2(cfg)
    model.logit_scale.data = np.array(np.log(a.init_scale))
    ln = FastLearner(model, TrainOpts(seed=a.seed, think_prob=0.0, use_stop=False,
                                      out_lr_mult=a.out_lr, debug_out=a.debug_out,
                                      track_preds=True))
    s = Stream(model)
    bits = np.empty(len(data))
    t0 = time.time()
    for i, b in enumerate(data):
        c = ln.step(s, b)
        bits[i] = np.nan if c is None else c / np.log(2)
    dt = time.time() - t0
    if a.ckpt:
        ln.save(a.ckpt)                              # simpan dulu: metrik tidak boleh menghilangkan hasil latih

    preds = np.array(ln.track_preds)
    preds = preds[:len(data) - 1]
    truth = np.frombuffer(data[1:1 + len(preds)], np.uint8)
    acc = float((preds == truth).mean())
    pc = np.bincount(preds, minlength=256)
    uniq = int((pc > 0).sum())
    pp = pc[pc > 0] / pc.sum()
    ent = float(-(pp * np.log2(pp)).sum())
    bm = bits[1:]
    third = len(bm) // 3

    sc = float(np.exp(model.logit_scale.data))
    bb = model.byte_bias.data.astype(np.float64)
    z = bb - bb.max()
    p_bias = np.exp(z) / np.exp(z).sum()
    ce_bias = float(-np.log2(np.maximum(p_bias[truth], 1e-300)).mean())
    tc = np.bincount(truth, minlength=256)
    ce_uni = float(-np.log2(np.maximum(tc / tc.sum(), 1e-300)[truth]).mean())

    def lab(x):
        return chr(x) if 32 <= x < 127 else f"0x{x:02x}"

    print(f"\n===== {a.label} =====")
    print(f"config: d={a.d} slow_every={a.slow_every} pcgrad={cfg.pcgrad} "
          f"head={cfg.head} separate_out={cfg.separate_out} out_lr={a.out_lr}")
    print(f"waktu {dt:.1f}s  {len(data)/dt:.0f} B/s  parameter={model.n_params()}")
    print(f"BPB semua           = {bm.mean():.3f}")
    print(f"BPB sepertiga akhir = {bm[-third:].mean():.3f}")
    print(f"akurasi argmax      = {acc*100:.2f}%")
    print(f"byte unik diprediksi= {uniq}/256")
    print(f"entropi prediksi    = {ent:.2f} bit   (unigram data {ce_uni:.2f} bit)")
    print(f"logit_scale         = {sc:.3f}   CE(softmax(byte_bias)) = {ce_bias:.3f} bpb")
    top = np.argsort(-pc)[:8]
    print("top prediksi: " + ", ".join(f"{lab(int(x))}:{pc[x]/pc.sum()*100:.0f}%" for x in top))

    if a.debug_out and ln.dbg:
        print("update awal (t, lr, wd, ||g|| parameter keluaran):")
        for t, lr, wd, g in ln.dbg[:10]:
            print(f"  t={t:>3} lr={lr:.4f} wd={wd:.3f}  " + "  ".join(f"{k}={v:.3e}" for k, v in g.items()))
        gs = {k: [] for k in ln.dbg_names}
        for _, _, _, g in ln.dbg:
            for k, v in g.items():
                gs[k].append(v)
        print("rata-rata ||g|| sepanjang latihan:  " +
              "  ".join(f"{k}={np.mean(v):.3e}" for k, v in gs.items() if v))

    print("SUMMARY|" + json.dumps({
        "label": a.label, "bpb": float(bm.mean()), "bpb_last3": float(bm[-third:].mean()),
        "acc": acc, "uniq": uniq, "ent": ent, "logit_scale": sc, "ce_bias": ce_bias, "sec": dt}))
    if a.ckpt:
        ln.save(a.ckpt)


if __name__ == "__main__":
    main()
