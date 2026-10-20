"""TINGKAT B — integrasi A+B pada skala yang diminta dokumen (>=1 jt byte).

Gerbang G2 (ARCHITECTURE_V2.md §7): "Tingkat B menambah >= +0,05 bit di atas Tingkat A pada
>=1M byte dan TIDAK mengecil di 10M". Skrip ini mengukur dengan protokol yang benar dan
KONTROL WAJIB:

  * A saja            — baseline.
  * A + B (terlatih)  — masukan neural dari Tingkat B terlatih.
  * A + B (acak)      — KONTROL: masukan neural dari Tingkat B BELUM TERLATIH (bobot acak).
  * A + B (kocok)     — KONTROL: masukan neural terlatih tapi urutannya DIKOCOK (alignment rusak).

Kalau kontrol "acak"/"kocok" ikut "menang", berarti protokolnya bocor (bukan neural yang berguna).
Ini pelajaran dari kebocoran ketiga (off-by-one) di proyek ini.

Jalan:
  python scripts/tierb_integrate.py --corpus data/indonesian_big.bin --ntrain 1000000 --ntest 50000
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from tmt2.gamelan.block import GamelanBlock
from tmt2.model import Config
from tmt2.stat_core import StatCore
from scripts.tierb_train import Adam, make_params, train_tierB, LN2
from scripts.m5_integrate import nn_stretch

# CATATAN: nn_stretch menyalin logika keselarasan dari m5_integrate (sudah diperbaiki dari
# kebocoran off-by-one). Impor ulang agar tidak divergen.


def tierb_bit_probs_random(blk, E, W_out, data, chunk=128):
    """Masukan neural dari Tingkat B ACAK (kontrol): bobot tak terlatih, tapi alur sama."""
    return nn_stretch(blk, E, W_out, data, chunk=chunk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/indonesian_big.bin")
    ap.add_argument("--ntrain", type=int, default=1_000_000)
    ap.add_argument("--ntest", type=int, default=50_000)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--conv_k", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--table_bits", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--jsonl", default="results/tierb_integrate.jsonl")
    a = ap.parse_args()

    with open(a.corpus, "rb") as f:
        raw = np.frombuffer(f.read(a.ntrain + a.ntest), dtype=np.uint8)
    if a.ntrain + a.ntest > raw.size:
        raise SystemExit(f"korpus {raw.size} B < {a.ntrain + a.ntest}")
    tr, te = raw[:a.ntrain], raw[a.ntrain:a.ntrain + a.ntest]
    print(f"KORPUS {a.corpus} · latih {len(tr):,} B · uji {len(te):,} B (ekor, tak beririsan)")
    print(f"Tingkat B d={a.d} H={a.heads} conv_k={a.conv_k} · {a.epochs} epoch · "
          f"Tier A tb={a.table_bits}\n", flush=True)

    # ---- [1] latih / muat Tingkat B
    if a.ckpt and os.path.exists(a.ckpt):
        d = np.load(a.ckpt, allow_pickle=True)
        meta = json.loads(str(d["meta"][0]))
        blk = GamelanBlock(d=int(meta["d"]), heads=int(meta["heads"]),
                           conv_k=int(meta["conv_k"]), seed=a.seed)
        for nm in ("wd", "bd", "wp", "wq", "wk", "wv", "wl", "bl", "wo", "wg", "bg",
                   "w1", "w2", "g1", "g2"):
            setattr(blk, nm, d[nm])
        E, W_out = d["E"], d["W_out"]
        print(f"[1] Tingkat B dimuat dari {a.ckpt} (BPB tercatat {meta['bpb']:.4f})", flush=True)
    else:
        print("[1] latih Tingkat B (teacher-forced)", flush=True)
        t0 = time.perf_counter()
        blk, E, W_out = train_tierB(tr, a.d, a.heads, a.conv_k, a.epochs, a.lr, a.batch, a.seed)
        print(f"    selesai {time.perf_counter()-t0:.0f}s", flush=True)

    # Tingkat B acak (kontrol): bobot baru, TIDAK dilatih
    blk_r = GamelanBlock(d=a.d, heads=a.heads, conv_k=a.conv_k, seed=999)
    _, E_r, W_out_r = make_params(blk_r, a.d, seed=999)

    # ---- [2] masukan neural (stretch) untuk latih & uji
    print("\n[2] hitung masukan neural (pohon biner)", flush=True)
    t0 = time.perf_counter()
    nn_tr = nn_stretch(blk, E, W_out, tr)
    nn_te = nn_stretch(blk, E, W_out, te)
    nn_tr_r = tierb_bit_probs_random(blk_r, E_r, W_out_r, tr)
    nn_te_r = tierb_bit_probs_random(blk_r, E_r, W_out_r, te)
    rng = np.random.default_rng(0)
    nn_tr_s = nn_tr[rng.permutation(len(nn_tr))]
    nn_te_s = nn_te[rng.permutation(len(nn_te))]
    print(f"    |st| rata-rata: terlatih {np.abs(nn_tr).mean():.3f} · "
          f"acak {np.abs(nn_tr_r).mean():.3f} ({time.perf_counter()-t0:.0f}s)", flush=True)

    # ---- [3] Tier A dengan/tanpa masukan neural
    print("\n[3] latih Tier A varian", flush=True)
    variants = [("A saja", None, None),
                ("A + B terlatih", nn_tr, nn_te),
                ("A + B acak", nn_tr_r, nn_te_r),
                ("A + B kocok", nn_tr_s, nn_te_s)]
    res = {}
    for tag, ntr, nte in variants:
        m = StatCore.from_config(Config.v2(stat_hash_bits=a.table_bits))
        for i, b in enumerate(tr):
            m.step(int(b), train=True, nn_st=(ntr[i] if ntr is not None else None))
        ce = m.eval_stream(te, nn_stream=(nte if nte is not None else None))
        # ⚠️ B-01 (diperbaiki 2026-10-06): StatCore.eval_stream mengembalikan CE dalam BIT
        # (stat_step memakai -math.log2). Versi lama membagi lagi dengan LN2 -> nilai
        # terinflasi x1,4427 (3,2159 seharusnya 2,2291). JANGAN bagi dengan LN2 di sini.
        res[tag] = float(ce.mean())
        print(f"    {tag:<16}: {res[tag]:.4f} bit", flush=True)

    base = res["A saja"]
    print("\n== PUTUSAN G2 ==")
    print(f"  A saja              : {base:.4f}")
    for tag in ("A + B terlatih", "A + B acak", "A + B kocok"):
        dlt = base - res[tag]
        print(f"  {tag:<19}: {res[tag]:.4f}  (Δ {dlt:+.4f})")
    d_tr = base - res["A + B terlatih"]
    d_rnd = base - res["A + B acak"]
    print(f"\n  Δ terlatih {d_tr:+.4f} vs ambang +0,05 -> "
          f"{'LULUS' if d_tr >= 0.05 else 'BELUM LULUS'}")
    if d_rnd > 0.005:
        print(f"  ⚠️ KONTROL ACAK juga 'menang' ({d_rnd:+.4f}) -> indikasi KEBOCORAN protokol.")
    else:
        print(f"  kontrol acak {d_rnd:+.4f} (≈0, protokol bersih).")

    with open(a.jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(corpus=a.corpus, ntrain=a.ntrain, ntest=a.ntest,
                                table_bits=a.table_bits, d=a.d, heads=a.heads,
                                epochs=a.epochs, **{k: v for k, v in res.items()})) + "\n")
    print(f"\nditulis -> {a.jsonl}")


if __name__ == "__main__":
    main()
