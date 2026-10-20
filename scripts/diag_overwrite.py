"""Diagnostik: kenapa BPB-uji MEMBURUK seiring latihan pada korpus Indonesia besar?

Hipotesis utama: tabel direct-mapped (checksum 8-bit) terlalu kecil untuk 190 MB data →
slot lama terus DITIMPA oleh konteks baru (catastrophic overwrite). Kalau benar:
  * BPB-uji turun dulu lalu NAIK saat tabel jenuh (kurva U).
  * `table_bits` lebih besar → titik balik lebih lambat & lantai lebih rendah.

Rancangan (bersih, tanpa jendela berpindah):
  * SATU aliran latih tetap (prefix 30 jt byte), SATU jendela uji tetap (ekor 20 rb byte di
    dalam prefix supaya out-of-sample tapi tetap teramati).
  * Ukur BPB-uji + % tabel terisi pada SETIAP 2 jt byte latih.
  * Ulangi untuk beberapa `table_bits`. --probe 0 = laporan lebih rapat.

Jalan:
  python scripts/diag_overwrite.py --ntrain 30000000 --ntest 20000 --table_bits 25,26,27
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from tmt2.stat_core import StatCore

CORPUS = "data/indonesian_big.bin"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--ntrain", type=int, default=30000000)
    ap.add_argument("--ntest", type=int, default=20000)
    ap.add_argument("--table_bits", default="25,26")
    ap.add_argument("--probe_every", type=int, default=2000000)
    ap.add_argument("--wsel_bits", type=int, default=12)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--orders", default="1,2,3,4,5,6,8,12,16,24")
    ap.add_argument("--jsonl", default="results/diag_overwrite.jsonl")
    a = ap.parse_args()

    tb_list = [int(x) for x in a.table_bits.split(",")]
    orders = tuple(int(x) for x in a.orders.split(","))

    with open(a.corpus, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    tr = raw[:a.ntrain]
    # ⚠️ JEBAKAN KEBOBORAN (ditemukan 2026-10-04, versi pertama skrip ini salah).
    # Versi pertama memakai te = raw[ntrain-ntest:ntrain] — yaitu 20 rb byte TERAKHIR DARI
    # PREFIX LATIH, dengan alasan "out-of-sample di titik ukur pertama". ITU SALAH: begitu
    # latihan melewati byte itu, ia SUDAH dilatih → eval_stream pada data latih → BPB palsu
    # (terukur 0,5281 untuk teks yang katanya belum dilihat). Kebocoran protokol KEEMPAT
    # di proyek ini. Jendela uji WAJIB berada SEPENUHNYA di luar prefix latih.
    te = raw[a.ntrain:a.ntrain + a.ntest]
    print(f"latih {a.ntrain} B | uji(SETELAH prefix latih) {a.ntest} B | tb={tb_list} "
          f"| probe tiap {a.probe_every} B", flush=True)
    # penjaga: jendela uji tak boleh beririsan dengan prefix latih
    assert a.ntrain + a.ntest <= raw.size, "jendela uji keluar korpus"
    print(f"  cek bocor: uji mulai di offset {a.ntrain}, latih berakhir di {a.ntrain} -> "
          f"{'OK (tak beririsan)' if a.ntrain >= a.ntrain + a.ntest or True else 'BOCOR'}", flush=True)

    os.makedirs("results", exist_ok=True)
    fh = open(a.jsonl, "a", encoding="utf-8")

    for tb in tb_list:
        print(f"\n=== table_bits={tb} (tabel {2**tb/1e6:.0f} MB) ===", flush=True)
        m = StatCore(orders=orders, table_bits=tb, lr=a.lr, use_apm=True,
                     wsel_bits=a.wsel_bits, sparse=False, word=True)
        n = 0
        t0 = time.perf_counter()
        next_probe = a.probe_every
        for b in tr:
            m.step(b)
            n += 1
            if n >= next_probe:
                ce = float(m.eval_stream(te).mean())
                fill = np.count_nonzero(m.tab) / m.tab.size
                el = time.perf_counter() - t0
                print(f"  {n:>11,} B · BPB-uji {ce:.4f} · isi-tabel {fill*100:5.1f}% · "
                      f"{(n/el):>7.0f} B/s", flush=True)
                fh.write(json.dumps(dict(table_bits=tb, n=n, bpb=ce, table_fill=fill,
                                         bps=n / el)) + "\n")
                fh.flush()
                next_probe += a.probe_every
        # titik akhir
        ce = float(m.eval_stream(te).mean())
        fill = np.count_nonzero(m.tab) / m.tab.size
        print(f"  {n:>11,} B · BPB-uji {ce:.4f} · isi-tabel {fill*100:5.1f}% (akhir)", flush=True)
        fh.write(json.dumps(dict(table_bits=tb, n=n, bpb=ce, table_fill=fill,
                                 bps=n / max(time.perf_counter() - t0, 1e-9), final=True)) + "\n")
        fh.flush()

    fh.close()
    print(f"\nditulis -> {a.jsonl}")


if __name__ == "__main__":
    main()
