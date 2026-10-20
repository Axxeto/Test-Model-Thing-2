"""E8 — KURVA TITIK-BEKU mixer+APM (memilih titik beku optimal untuk fix kurva-U).

Latar:
  E7 (HASIL_E7_PEMISAHAN.md): kerusakan kurva-U dibawa oleh MIXER (cangkok mixer@3jt memulihkan
  89%); APM bukan sebab. E1: makin awal mixer/APM dibekukan makin baik (3jt 2,1456 · 9jt 2,1675 ·
  18jt 2,3756).
  Karena update TABEL independen dari bobot mixer, "latih 30 jt dengan mixer/APM dibekukan di T"
  menghasilkan state yang SAMA dengan cangkok `tab@30jt + mixer/APM@T`. Jadi kurva titik-beku bisa
  diukur murah: SATU latih 30 jt, snapshot mixer/APM murah (~20 MB) di tiap T, lalu cangkok.

Protokol: korpus data/indonesian_big.bin, tb=25, suite uji TETAP 20x200 kB i.i.d., satuan CE = BIT.

Jalan:
  python scripts/e8_freeze_curve.py --ntrain 30000000 --table_bits 25 \
      --snap_at 1000000,3000000,6000000,9000000,18000000 --jsonl results/e8_freeze_curve.jsonl
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from tmt2.stat_core import StatCore


def build_core(a):
    orders = tuple(int(x) for x in a.orders.split(","))
    return StatCore(orders=orders, table_bits=a.table_bits, wsel_bits=a.wsel_bits,
                    cks_bits=a.cks_bits, match_fix=True, ctx_fix=True, use_apm=True)


def snap_cheap(m):
    """Snapshot MURAH: hanya mixer+APM (tanpa tabel raksasa)."""
    return dict(w1=m.w1.copy(), w2=m.w2.copy(), apm1=m.apm1.copy(), apm2=m.apm2.copy())


def graft_eval(a, tab, cheap, windows):
    """CE rata-rata pada suite; tab dari state final, mixer/APM dari titik beku T."""
    per = []
    for w in windows:
        m = build_core(a)
        m.tab[:] = tab
        m.w1[:] = cheap["w1"]
        m.w2[:] = cheap["w2"]
        m.apm1[:] = cheap["apm1"]
        m.apm2[:] = cheap["apm2"]
        per.append(float(np.nanmean(m.eval_stream_fast(w))))
    return float(np.mean(per)), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/indonesian_big.bin")
    ap.add_argument("--ntrain", type=int, default=30_000_000)
    ap.add_argument("--table_bits", type=int, default=25)
    ap.add_argument("--wsel_bits", type=int, default=12)
    ap.add_argument("--cks_bits", type=int, default=24)
    ap.add_argument("--orders", default="1,2,3,4,5,6,8,12,16,24")
    ap.add_argument("--snap_at", default="1000000,3000000,6000000,9000000,18000000")
    ap.add_argument("--nwin", type=int, default=20)
    ap.add_argument("--win", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--jsonl", default="results/e8_freeze_curve.jsonl")
    a = ap.parse_args()

    snap_at = sorted(int(x) for x in a.snap_at.split(",") if x.strip())
    need = a.ntrain + a.nwin * a.win
    with open(a.corpus, "rb") as f:
        raw = np.frombuffer(f.read(need), dtype=np.uint8)
    if len(raw) < need:
        raise SystemExit(f"korpus kurang: {len(raw)} < {need}")
    tr = raw[:a.ntrain]
    test_region = raw[a.ntrain:]
    rng = np.random.default_rng(a.seed)
    starts = rng.choice(len(test_region) - a.win, size=a.nwin, replace=False)
    windows = [test_region[s:s + a.win] for s in starts]

    print(f"[E8] tb={a.table_bits} latih {a.ntrain} byte · titik beku {snap_at} · "
          f"suite {a.nwin}x{a.win}B i.i.d.", flush=True)
    m = build_core(a)
    cheap = {}
    t0 = time.perf_counter()
    prev = 0
    for nb in sorted(set(snap_at) | {a.ntrain}):
        nb = min(nb, a.ntrain)
        if nb > prev:
            m.train_stream(tr[prev:nb])
            prev = nb
        el = time.perf_counter() - t0
        print(f"  latih sampai {prev/1e6:.1f} jt byte ({el:.0f}s, {prev/max(el,1e-9):.0f} B/s)",
              flush=True)
        if prev in snap_at:
            cheap[prev] = snap_cheap(m)
            print(f"    -> snapshot mixer/APM @ {prev/1e6:.1f} jt", flush=True)
    tab_final = m.tab.copy()

    # titik beku: T dari daftar + model utuh (T = ntrain)
    points = list(cheap.keys()) + [a.ntrain]
    cheap[a.ntrain] = snap_cheap(m)
    print("\n[E8] evaluasi (tab@30jt tetap, mixer/APM dibekukan di T):", flush=True)
    rows = []
    for T in points:
        bpb, per = graft_eval(a, tab_final, cheap[T], windows)
        rows.append(dict(T=T, bpb=bpb, sd=float(np.std(per))))
        print(f"  beku@{T/1e6:5.1f}jt: {bpb:.4f} bit  (sd {np.std(per):.4f})", flush=True)

    best = min(rows, key=lambda r: r["bpb"])
    full = next(r for r in rows if r["T"] == a.ntrain)
    se = full["sd"] / max(a.nwin, 1) ** 0.5
    print("\n== KEPUTUSAN E8 ==", flush=True)
    print(f"  utuh (tanpa beku) : {full['bpb']:.4f} bit", flush=True)
    print(f"  TERBAIK: beku@{best['T']/1e6:.1f}jt -> {best['bpb']:.4f} bit "
          f"(Δ {full['bpb']-best['bpb']:+.4f})", flush=True)
    print(f"  SE antarsjendela = {se:.4f} (ambang 2xSE = {2*se:.4f})", flush=True)

    with open(a.jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(corpus=a.corpus, ntrain=a.ntrain, table_bits=a.table_bits,
                                nwin=a.nwin, win=a.win, snap_at=snap_at, se=se,
                                best_T=best["T"], best_bpb=best["bpb"], rows=rows)) + "\n")
    print(f"\nditulis -> {a.jsonl}")


if __name__ == "__main__":
    main()
