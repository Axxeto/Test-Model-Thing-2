"""Latih Tingkat A pada korpus Indonesia 1M dan hasilkan teks.

Ini jalur TERBAIK & TERCEPAT yang tersedia:
  * Tingkat A (inti statistik biner) — BPB tahan Indonesia **2,3261** pada 200k/8k,
    mengalahkan TMT2+mixer simbol (2,4809), n-gram JM-5 (2,5400), dan **bz2 -9 (2,3700)**,
    pada ~100k B/s (≈35× lebih cepat dari jalur simbol 2.666 B/s).
  * Konfigurasi produksi: `table_bits=25`, `wsel_bits=12`, APM on, match on, word on,
    sparse OFF (terbukti merugi +0,038 bit).

Skrip ini: (1) latih pada SELURUH 965 kB, (2) laporkan BPB pada jendela uji tetap,
(3) hasilkan beberapa sampel dari prompt berbeda pada beberapa suhu.

Jalan:
  python scripts/train_gen_id.py --gen 700 --temp 0.75
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from tmt2.corpus import load
from tmt2.stat_core import StatCore

CORPUS = "data/indonesian_1m_pure.txt"

PROMPTS = [
    "Indonesia adalah negara kepulauan yang",
    "Pada tahun 1945, para pendiri bangsa",
    "Kucing merupakan hewan mamalia yang",
    "Teknologi informasi telah mengubah cara",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ntrain", type=int, default=0, help="0 = pakai seluruh korpus minus uji")
    ap.add_argument("--ntest", type=int, default=8000)
    ap.add_argument("--table_bits", type=int, default=25)
    ap.add_argument("--wsel_bits", type=int, default=12)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--orders", default="1,2,3,4,5,6,8,12,16,24")
    ap.add_argument("--no_apm", action="store_true")
    ap.add_argument("--no_sparse", action="store_true", default=True,
                    help="sparse terbukti MERUGI (+0,038 bit) → default mati")
    ap.add_argument("--sparse", action="store_true", help="nyalakan konteks jarang (ablasi)")
    ap.add_argument("--gen", type=int, default=700, help="jumlah byte yang dihasilkan")
    ap.add_argument("--temp", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/gen_id_tierA.txt")
    a = ap.parse_args()

    orders = tuple(int(x) for x in a.orders.split(","))
    raw = np.frombuffer(load(CORPUS), dtype=np.uint8)
    total = raw.size
    te = raw[total - a.ntest:]
    ntrain = a.ntrain if a.ntrain > 0 else total - a.ntest
    tr = raw[:ntrain]

    print(f"KORPUS Indonesia · {total} byte · latih {ntrain} · uji {a.ntest} "
          f"(jendela tetap = ekor korpus)")
    print(f"Tingkat A · table_bits={a.table_bits} · wsel_bits={a.wsel_bits} · "
          f"APM={'off' if a.no_apm else 'on'} · orders={orders}")

    t0 = time.perf_counter()
    m = StatCore(orders=orders, table_bits=a.table_bits, lr=a.lr,
                 use_apm=not a.no_apm, wsel_bits=a.wsel_bits,
                 sparse=a.sparse, word=True)
    warm = 4000
    for b in tr[:warm]:
        m.step(b)
    tw = time.perf_counter()
    for b in tr[warm:]:
        m.step(b)
    dt = time.perf_counter() - tw
    bps = (len(tr) - warm) / dt
    tt = time.perf_counter() - t0
    print(f"latih selesai: {tt:.1f} s · throughput ~{bps:.0f} B/s (steadi-state)")

    ce = float(m.eval_stream(te).mean())
    print(f"BPB tahan (jendela uji {a.ntest} B): {ce:.4f}")

    # ---- generasi
    lines = []
    lines.append(f"# Generasi Tingkat A — korpus Indonesia 1M")
    lines.append(f"# latih {ntrain} B · BPB tahan {ce:.4f} · temp {a.temp} · "
                 f"table_bits {a.table_bits} · wsel {a.wsel_bits}")
    lines.append("")
    for i, p in enumerate(PROMPTS):
        # tiap prompt pakai SALINAN keadaan: latih ulang cepat? Tidak — generate() hanya
        # membaca konteks prompt lalu mengumpankannya; tabel tetap beku. Aman berbagi.
        for s in range(2):
            txt = m.generate(p, a.gen, temp=a.temp, seed=a.seed + i * 10 + s)
            try:
                dec = txt.decode("utf-8", errors="replace")
            except Exception:
                dec = repr(txt)
            lines.append(f"=== prompt: {p!r}  (seed {a.seed + i * 10 + s}) ===")
            lines.append(p + dec)
            lines.append("")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nditulis ke {a.out}")
    with open("results/gen_id_tierA.jsonl", "a") as f:
        f.write(json.dumps(dict(ntrain=ntrain, ntest=a.ntest, bpb=ce, bps=bps,
                                table_bits=a.table_bits, wsel_bits=a.wsel_bits,
                                temp=a.temp, gen=a.gen, seed=a.seed)) + "\n")
    # cetak cuplikan ke layar
    print("\n---- cuplikan generasi ----")
    for i, p in enumerate(PROMPTS[:2]):
        txt = m.generate(p, 300, temp=a.temp, seed=a.seed + i * 10)
        print(f"\n[{p}]")
        print((p + txt.decode("utf-8", errors="replace"))[:400])


if __name__ == "__main__":
    main()
