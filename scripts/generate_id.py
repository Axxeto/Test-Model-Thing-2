"""Latih TMT2 pada teks Indonesia lalu GENERASI contoh keluaran (byte-level).

Tujuan: uji apakah model benar-benar bisa menghasilkan teks, bukan sekadar menurunkan BPB.

  python scripts/generate_id.py --data data/indonesian_1m_pure.txt --n 300000 \
         --train_from 0 --d 32 --mixer --steps 3 --out_id results/gen_id.txt

Protokol:
  - Latih ONLINE pada prefix data (prekuensial, tanpa kebocoran: prediksi byte t sebelum lihat t).
  - Generasi dengan prompt dari data LATIH (model boleh "mengingat"), plus prompt bebas.
  - Beberapa suhu: rendah (0,5) -> konservatif/repetitif; tinggi (1,0) -> liar.
  - Bandingkan dua varian: mixer aktif (Config.best) vs SSM-saja, supaya terlihat apa yang
    disumbang mixer pada GENERASI (bukan hanya BPB).
"""
import argparse, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tmt2.model import Config, TMT2
from tmt2.learner import TrainOpts, Stream
from tmt2.fast_learner import FastLearner

LN2 = float(np.log(2))


def build(cfg_kw, opts_kw, seed):
    cfg_kw = dict(cfg_kw)
    cfg_kw["seed"] = seed                      # jangan dobel: buang seed dari cfg_kw dulu
    cfg = Config(**cfg_kw)
    m = TMT2(cfg)
    fl = FastLearner(m, TrainOpts(seed=seed, **opts_kw))
    return m, fl


def train(fl, data, nbytes, report=50000):
    s = Stream(fl.m, online=True)
    t0 = time.time()
    ce, n = 0.0, 0
    for i, b in enumerate(data[:nbytes]):
        c = fl.step(s, int(b), train=True, think=False)
        if c is not None:
            ce += float(c); n += 1
        if report and (i + 1) % report == 0:
            print(f"    [{i+1:>7d}] CE kumulatif {ce/max(n,1)/LN2:.4f} bpB  "
                  f"({(i+1)/(time.time()-t0):.0f} B/s)", flush=True)
    return ce / max(n, 1) / LN2, time.time() - t0


def show(fl, prompt, n, temps, seed=0):
    """Cetak generasi untuk beberapa suhu. Prompt = bytes."""
    print(f"    prompt: {prompt.decode('utf-8','replace')!r}")
    for T in temps:
        gen = fl.generate(prompt, n, temp=T, think=False, seed=seed)
        txt = gen.decode("utf-8", "replace") if isinstance(gen, (bytes, bytearray)) else str(gen)
        print(f"      T={T:<4} -> {txt!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/indonesian_1m_pure.txt")
    ap.add_argument("--n", type=int, default=300000, help="byte latih")
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--gen_len", type=int, default=200)
    ap.add_argument("--temps", default="0.5,0.7,0.9")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_mixer", action="store_true", help="matikan mixer (SSM-saja) sbg pembanding")
    ap.add_argument("--out_id", default="results/gen_id.txt")
    a = ap.parse_args()

    raw = open(a.data, "rb").read()
    data = raw[:a.n]
    print(f"data: {a.data}  latih {len(data)} byte  uniq {len(set(data))}")

    if a.no_mixer:
        cfg_kw = dict(vocab=256, d=a.d, n_layers=a.layers, d_mem=max(8, a.d // 4),
                      head="linear", deep_grad=True, lr=1e-2, warmup=200, slow_every=8,
                      seg_len=256, replay=False, sleep_every=10 ** 9, mixer_k=0)
        tag = "SSM-saja (mixer mati)"
    else:
        cfg_kw = Config.best(vocab=256, d=a.d, n_layers=a.layers, d_mem=max(8, a.d // 4),
                             slow_every=8, seg_len=256, replay=False, sleep_every=10 ** 9).to_dict()
        tag = "Config.best (mixer orde 1..8)"
    opts_kw = dict(think_prob=0.0, only_byte_loss=True, use_stop=False)
    m, fl = build(cfg_kw, opts_kw, a.seed)
    print(f"\n=== {tag} ===  params={m.n_params()}")

    bpb, secs = train(fl, data, len(data))
    print(f"  -> CE prekuensial {bpb:.4f} bpB  ({secs:.0f}s)")

    temps = [float(t) for t in a.temps.split(",")]
    lines = [f"=== TMT2 generasi Indonesia | {tag} | latih {len(data)}B | CE {bpb:.4f} bpB ===", ""]
    for src_name, prompt in [("awal data", data[:60]),
                             ("tengah data", data[len(data)//2:len(data)//2+60]),
                             ("prompt bebas", b"Pada suatu hari di Jakarta,"),
                             ("prompt bebas 2", b"Ibu pergi ke pasar untuk membeli")]:
        lines.append(f"-- {src_name} --")
        print(f"\n  -- {src_name} --")
        for T in temps:
            gen = fl.generate(prompt, a.gen_len, temp=T, think=False, seed=a.seed)
            txt = gen.decode("utf-8", "replace") if isinstance(gen, (bytes, bytearray)) else str(gen)
            cont = txt[len(prompt.decode("utf-8","replace")):] if txt.startswith(prompt.decode("utf-8","replace")) else txt
            print(f"    T={T:<4} -> {txt!r}")
            lines.append(f"  T={T}  {src_name}: {txt}")

    os.makedirs(os.path.dirname(a.out_id), exist_ok=True)
    with open(a.out_id, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n-> {a.out_id}")


if __name__ == "__main__":
    main()
