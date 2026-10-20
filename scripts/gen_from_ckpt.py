"""Muat model Tingkat A dari checkpoint .npz lalu hasilkan teks (mode inferensi beku).

Bukti bahwa model dapat dibangun-ulang dari checkpoint dan dipakai mandiri.

Jalan:
  python scripts/gen_from_ckpt.py --ckpt checkpoints/id_big_30m.npz \
      --prompt "Indonesia adalah negara kepulauan yang" --gen 500 --temp 0.6
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from tmt2.stat_core import StatCore


def load_model(path):
    # Delegasi ke StatCore.load (sumber kebenaran tunggal format ckpt, §11.3).
    return StatCore.load(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/id_big_30m.npz")
    ap.add_argument("--prompt", default="Indonesia adalah negara kepulauan yang")
    ap.add_argument("--gen", type=int, default=500)
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    m, meta = load_model(a.ckpt)
    print(f"model dimuat: latih {meta['ntrain']:,} B · BPB tercatat {meta['bpb']:.4f} · "
          f"tb={meta['table_bits']}")
    txt = m.generate(a.prompt, a.gen, temp=a.temp, seed=a.seed)
    print(f"\nprompt: {a.prompt}")
    print("-" * 60)
    print(a.prompt + txt.decode("utf-8", errors="replace"))


if __name__ == "__main__":
    main()
