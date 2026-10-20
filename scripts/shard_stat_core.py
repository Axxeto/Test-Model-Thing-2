"""M6 — shard multi-inti Tingkat A: skala throughput + kualitas campuran.

Struktur: N inti, masing-masing satu StatCore independen. Latihan: data dibagi rata jadi N
bagian yang SALING LEPAS (mode ``split``) atau setiap inti melatih SELURUH data (mode
``full``). Penilaian: tiap inti memprediksi jendela uji yang SAMA, probabilitas per-bit
dirata-ratakan (campuran aritmetik), lalu BPB dihitung dari campuran itu.

Dua pertanyaan yang dijawab:
  1. KECEPATAN — apakah agregat N inti ≈ N× satu inti? (G3: ≥500k B/s agregat.)
  2. KUALITAS — apakah rata-rata N tabel LEBIH BAIK atau LEBIH BURUK dari satu tabel?

HASIL (WikiText 200k/8k, table_bits=22, wsel_bits=12) — lihat results/shard_*.jsonl:
  * ``split``: BPB 2,44 → 6,14 → 12,16 → 21,31 (N=1,2,4,8) — **kualitas RUNTUH**.
  * ``full`` : BPB 2,44 → 5,99 → 11,55 → 19,35, tapi kecepatan naik sampai **4,62×** @8 inti.
  * Penyebab: inti biner sangat JENUH (16,5% bit di-clamp ke 1e-6/1−1e-6) dan DETERMINISTIK
    (dua model dengan lr berbeda memberi BPB identik 2,5077/2,5076) → tak ada keragaman, dan
    merata-ratakan prediksi yang saling bertentangan justru menghukum keduanya.
  * **Kesimpulan jujur**: shard = pengungkit KECEPATAN penilaian, BUKAN kualitas. Untuk
    kualitas, satu tabel pada semua data tetap yang terbaik.

⚠️ **Jebakan satuan**: rata-rata CE harus dihitung per BYTE (×8), bukan per bit. Versi pertama
melaporkan 0,315 — itu CE/bit, bukan BPB; BPB yang benar = 2,522 (rasio tepat 8×). Bukan
kebocoran, hanya lupa faktor 8.

Jalan:
  python scripts/shard_stat_core.py --nshard 1,2,4,8 --ntrain 200000 --mode full
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from tmt2.corpus import load
from tmt2.stat_core import StatCore

CORPORA = {"wikitext": "wikitext", "indonesian": "data/indonesian_1m_pure.txt"}

_G = {}


def _worker_init(kind, ntrain, ntest, table_bits, lr, orders, apm, wsel, sparse, word, mode):
    raw = np.frombuffer(load(CORPORA[kind]), dtype=np.uint8)
    total = raw.size
    te = raw[total - ntest:]
    tr = raw[:min(ntrain, total - ntest)]
    _G.update(tr=tr, te=te, table_bits=table_bits, lr=lr, orders=orders,
              apm=apm, wsel=wsel, sparse=sparse, word=word, mode=mode)


def _train_shard(args):
    """Latih satu shard, kembalikan probabilitas p(bit=1) pada jendela uji (panjang 8×ntest).

    ``mode="split"``: shard i melatih pada potongan ke-i (data LEPAS, total kerja tetap).
    ``mode="full"``:  setiap shard melatih pada SELURUH data (ansambel, kerja ×N tapi tiap
    anggota melihat semua konteks — satu-satunya mode yang boleh diklaim menambah kualitas).
    """
    idx, nshard = args
    tr = _G["tr"]
    mode = _G.get("mode", "split")
    if mode == "split":
        seg = np.array_split(tr, nshard)[idx]
    else:
        seg = tr
    m = StatCore(orders=_G["orders"], table_bits=_G["table_bits"], lr=_G["lr"],
                 use_apm=_G["apm"], wsel_bits=_G["wsel"], sparse=_G["sparse"], word=_G["word"])
    nwarm = min(4000, len(seg))
    for b in seg[:nwarm]:
        m.step(b)
    t0 = time.perf_counter()
    for b in seg[nwarm:]:
        m.step(b)
    dt = time.perf_counter() - t0
    te = _G["te"]
    P = np.empty(len(te) * 8, dtype=np.float64)
    k = 0
    for j, b in enumerate(te):
        c = m.step(b, train=False)
        pr = m._probs
        P[k:k + 8] = pr
        k += 8
        if c is None:                     # byte pertama (hlen==0): _probs belum ditulis
            P[k - 8:k] = 0.5
    return idx, P, len(seg) - nwarm, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="wikitext")
    ap.add_argument("--nshard", default="1,2,4,8")
    ap.add_argument("--ntrain", type=int, default=200000)
    ap.add_argument("--ntest", type=int, default=8000)
    ap.add_argument("--table_bits", type=int, default=22)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--orders", default="1,2,3,4,5,6,8,12,16,24")
    ap.add_argument("--wsel_bits", type=int, default=12)
    ap.add_argument("--mode", choices=["split", "full"], default="split",
                    help="split: tiap shard melatih 1/N data (kerja tetap) · "
                         "full: tiap shard melatih SELURUH data (ansambel, kerja ×N)")
    ap.add_argument("--out", default="results/shard_stat_core.jsonl")
    a = ap.parse_args()

    orders = tuple(int(x) for x in a.orders.split(","))
    shards = [int(x) for x in a.nshard.split(",")]
    raw = np.frombuffer(load(CORPORA[a.corpus]), dtype=np.uint8)
    total = raw.size
    te = raw[total - a.ntest:]
    bits = np.unpackbits(te)               # bit sebenarnya, urut MSB->LSB per byte (cocok kernel)
    nbit = len(bits)

    print(f"M6 SHARD · korpus={a.corpus} · ntrain={a.ntrain} ntest={a.ntest} "
          f"table_bits={a.table_bits} · mode={a.mode}")
    print(f"{'shard':>6}{'BPB campuran':>14}{'B/s agregat':>14}{'B/s/inti':>11}{'speedup':>9}")
    print("-" * 54)
    rows = []
    base_single = None
    for n in shards:
        t0 = time.perf_counter()
        with Pool(n, initializer=_worker_init,
                  initargs=(a.corpus, a.ntrain, a.ntest, a.table_bits, a.lr, orders,
                            True, a.wsel_bits, False, True, a.mode)) as pool:
            res = pool.map(_train_shard, [(i, n) for i in range(n)])
        wall = time.perf_counter() - t0
        P = np.ones(nbit, dtype=np.float64) / n
        tot_bytes = 0
        for idx, Pp, nb, dt in res:
            P *= Pp
            tot_bytes += nb
        P = np.clip(P, 1e-12, 1 - 1e-12)
        # CE per BIT lalu dikali 8 -> BPB. (Jangan lupa faktor 8: rata-rata atas bit, bukan byte.)
        per_bit = -(bits * np.log2(P) + (1 - bits) * np.log2(1 - P))
        ce = float(per_bit.mean()) * 8.0
        agg_bps = tot_bytes / wall if wall > 0 else float("nan")
        per_core = agg_bps / n
        if base_single is None:
            base_single = agg_bps
        speed = agg_bps / base_single
        print(f"{n:>6}{ce:>14.4f}{agg_bps:>14.0f}{per_core:>11.0f}{speed:>8.2f}x", flush=True)
        rows.append(dict(corpus=a.corpus, nshard=n, mode=a.mode, bpb=float(ce),
                         agg_bps=agg_bps, per_core_bps=per_core, speedup=speed,
                         ntrain=a.ntrain, ntest=a.ntest))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nditulis ke {a.out}")
    print(f"Catatan: mode={a.mode}. 'full' = ansambel (tiap anggota lihat semua data, kerja ×N); "
          "'split' = data dibagi (kerja tetap, tiap anggota lihat 1/N data).")


if __name__ == "__main__":
    main()
