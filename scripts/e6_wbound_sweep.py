"""E6 — sapuan BATAS BOBOT mixer (fix kurva-U).

Latar: E1+E2 (lihat HASIL_E1_E2_KURVA_U.md) membuktikan kurva-U disebabkan bobot mixer lapis-1
yang divergen (max|w1| 1,39 -> 1676 sepanjang 30 jt byte). Skrip ini menyapu `w_bound` (clip
|bobot| = ±w_bound setelah tiap pembaruan) pada protokol BERSIH: latih N byte, evaluasi pada
jendela uji TETAP di luar latih.

Pertanyaan yang dijawab: apakah membatasi bobot menghapus/menunda kurva-U (BPB-uji tak lagi naik)?

Jalan:
  python scripts/e6_wbound_sweep.py --ntrain 30000000 --table_bits 25 --bounds 0,2,4,8,16
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from tmt2.stat_core import StatCore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/indonesian_big.bin")
    ap.add_argument("--ntrain", type=int, default=30_000_000)
    ap.add_argument("--ntest", type=int, default=200_000)
    ap.add_argument("--table_bits", type=int, default=25)
    ap.add_argument("--wsel_bits", type=int, default=12)
    ap.add_argument("--cks_bits", type=int, default=24)
    ap.add_argument("--orders", default="1,2,3,4,5,6,8,12,16,24")
    ap.add_argument("--bounds", default="0,2,4,8,16",
                    help="daftar w_bound (0 = tanpa batas = perilaku lama)")
    ap.add_argument("--jsonl", default="results/e6_wbound_sweep.jsonl")
    a = ap.parse_args()

    orders = tuple(int(x) for x in a.orders.split(","))
    bounds = [float(x) for x in a.bounds.split(",") if x.strip()]
    need = a.ntrain + a.ntest
    with open(a.corpus, "rb") as f:
        raw = np.frombuffer(f.read(need), dtype=np.uint8)
    tr, te = raw[:a.ntrain], raw[a.ntrain:]

    print(f"[E6] tb={a.table_bits} ntrain={a.ntrain} bounds={bounds}", flush=True)
    rows = []
    for wb in bounds:
        m = StatCore(orders=orders, table_bits=a.table_bits, wsel_bits=a.wsel_bits,
                     cks_bits=a.cks_bits, match_fix=True, ctx_fix=True, use_apm=True,
                     w_bound=wb)
        t0 = time.perf_counter()
        m.train_stream(tr)
        el = time.perf_counter() - t0
        bpb = float(np.nanmean(m.eval_stream_fast(te)))
        w1mx = float(np.max(np.abs(m.w1)))
        w1mn = float(np.mean(np.abs(m.w1)))
        rec = dict(w_bound=wb, bpb=bpb, w1_max=w1mx, w1_mean=w1mn,
                   bps=round(len(tr) / max(el, 1e-9), 0))
        rows.append(rec)
        print(f"  w_bound={wb:5.1f}: BPB={bpb:.4f}  max|w1|={w1mx:9.3f}  "
              f"mean|w1|={w1mn:.4f}  ({rec['bps']:.0f} B/s)", flush=True)
        # ★ Tulis TIAP bound begitu selesai -> pembatalan di tengah tak menghapus hasil sebelumnya.
        with open(a.jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(kind="row", corpus=a.corpus, ntrain=a.ntrain,
                                    table_bits=a.table_bits, orders=list(orders),
                                    row=rec)) + "\n")

    # verdict sederhana: mana yang memberi BPB terbaik
    best = min(rows, key=lambda r: r["bpb"])
    base = next((r for r in rows if r["w_bound"] == 0.0), None)
    print("\n== KEPUTUSAN E6 ==")
    if base:
        print(f"  tanpa batas (0)  : BPB {base['bpb']:.4f}")
    print(f"  TERBAIK: w_bound={best['w_bound']:.1f} -> BPB {best['bpb']:.4f}")
    if base and best["w_bound"] != 0.0:
        d = base["bpb"] - best["bpb"]
        print(f"  perbaikan vs tanpa batas: {d:+.4f} bit "
              f"({'MEMBANTU' if d > 0.005 else 'tidak jelas' if abs(d) <= 0.005 else 'MERUGIKAN'})")

    with open(a.jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(kind="summary", corpus=a.corpus, ntrain=a.ntrain,
                                table_bits=a.table_bits, orders=list(orders),
                                best_w_bound=best["w_bound"], best_bpb=best["bpb"],
                                rows=rows)) + "\n")
    print(f"\nditulis -> {a.jsonl}")


if __name__ == "__main__":
    main()
