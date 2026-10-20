"""E7b — CANGKOK PER-LAPIS & PER-SET (Fase 0, zero-risk, tanpa ubah perilaku).

Melokalisasi pembawa kurva-U di dalam {w1, w2, apm}: mencangkok w1 SAJA, w2 SAJA,
atau satu set w1[k] (k=0,1,2) dari titik awal S ke model akhir L, memakai
`StatCore.eval_suite` (tanpa salin tabel 268MB per jendela) + `diagnose()` untuk
fraksi jenuh / CE-sendiri / w2neg / reliabilitas (uji M1/M2).

Prediksi M1: `w1@S` memulihkan ~penuh, terkonsentrasi 1-2 set; `w2@S` kecil efeknya.

Jalan kecil (validasi tooling, ~2 menit):
  python scripts/e7b_graft_layer.py --ntrain 3000000 --snap_at 1000000,3000000 \
      --table_bits 22 --nwin 8 --win 50000
Jalan penuh (verdict, ~10 menit):
  python scripts/e7b_graft_layer.py --ntrain 30000000 --snap_at 3000000,30000000 \
      --table_bits 25 --nwin 20 --win 200000
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from tmt2.console import ensure_utf8
from tmt2.stat_core import StatCore

ensure_utf8()


def build(a):
    orders = tuple(int(x) for x in a.orders.split(","))
    return StatCore(orders=orders, table_bits=a.table_bits, wsel_bits=a.wsel_bits,
                    cks_bits=a.cks_bits, match_fix=True, ctx_fix=True, use_apm=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/indonesian_big.bin")
    ap.add_argument("--ntrain", type=int, default=3_000_000)
    ap.add_argument("--table_bits", type=int, default=22)
    ap.add_argument("--wsel_bits", type=int, default=12)
    ap.add_argument("--cks_bits", type=int, default=24)
    ap.add_argument("--orders", default="1,2,3,4,5,6,8,12,16,24")
    ap.add_argument("--snap_at", default="1000000,3000000")
    ap.add_argument("--nwin", type=int, default=8)
    ap.add_argument("--win", type=int, default=50_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--jsonl", default="results/e7b_graft_layer.jsonl")
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
    windows = [test_region[s:s + a.win].copy() for s in starts]

    print(f"[E7b] tb={a.table_bits} latih {a.ntrain} B, snapshot {snap_at}", flush=True)
    m = build(a)
    snaps, tab_at = {}, {}
    t0 = time.perf_counter()
    prev = 0
    for nb in sorted(set(snap_at) | {a.ntrain}):
        nb = min(nb, a.ntrain)
        if nb > prev:
            m.train_stream(tr[prev:nb])
            prev = nb
        el = time.perf_counter() - t0
        print(f"  latih {prev/1e6:.1f}jt ({el:.0f}s, {prev/max(el,1e-9):.0f} B/s)", flush=True)
        if prev in snap_at:
            snaps[prev] = m.mix_snapshot()
            tab_at[prev] = m.tab_snapshot()
            print(f"    snapshot @{prev/1e6:.1f}jt (max|w1|={np.abs(m.w1).max():.2f})", flush=True)
    S, L = snap_at[0], snap_at[-1]

    # --- diagnostik M1/M2 pada S dan L (satu model, tabel akhir, mixer diganti) ---
    for T, tag in ((S, "S"), (L, "L")):
        g = build(a)
        g.load_tab(tab_at[L])
        g.load_mix(snaps[T])
        d = g.diagnose(windows[0][:20000])
        print(f"  diag mix@{tag}: sat={['%.3f' % s for s in d['sat_frac']]} "
              f"ownCE={[round(v*8,3) for v in d['own_ce']]} pfCE={d['pf_ce']*8:.4f} "
              f"w2neg={d['w2neg_frac']:.3f} w1max={d['w1max']:.1f} "
              f"argmax={d['w1argmax']}", flush=True)

    # --- graft ---
    res, sd = {}, {}

    def run(label, tab, mix):
        g = build(a)
        g.load_tab(tab)
        g.load_mix(mix)
        bpb, per = g.eval_suite(windows, cold=True)
        res[label] = bpb
        sd[label] = float(np.std(per))
        print(f"  {label:28s}: {bpb:.4f} bit (sd {np.std(per):.4f})", flush=True)
        return bpb

    print("[E7b] evaluasi:", flush=True)
    uS = run(f"utuh@{S//1000000}jt", tab_at[S], snaps[S])
    uL = run(f"utuh@{L//1000000}jt", tab_at[L], snaps[L])

    def mix_with(w1=None, w2=None, apm_from=None, w1sets=None):
        base = {k: v.copy() for k, v in snaps[L].items()}
        if w1 is not None:
            base["w1"] = snaps[w1]["w1"].copy()
        if w1sets is not None:  # dict k -> snap time
            base["w1"] = snaps[L]["w1"].copy()
            for k, t in w1sets.items():
                base["w1"][k] = snaps[t]["w1"][k].copy()
        if w2 is not None:
            base["w2"] = snaps[w2]["w2"].copy()
        if apm_from is not None:
            base["apm1"] = snaps[apm_from]["apm1"].copy()
            base["apm2"] = snaps[apm_from]["apm2"].copy()
        return base

    run(f"tabL+w1@{S//1000000}+restoL", tab_at[L], mix_with(w1=S))
    run(f"tabL+w2@{S//1000000}+restoL", tab_at[L], mix_with(w2=S))
    for k in range(3):
        run(f"tabL+w1set{k}@{S//1000000}", tab_at[L], mix_with(w1sets={k: S}))
    run(f"tabL+all@{S//1000000}(kontrol)", tab_at[L], snaps[S])
    run(f"tab@{S//1000000}+all@{L//1000000}(kontrol)", tab_at[S], snaps[L])

    se = sd[f"utuh@{L//1000000}jt"] / max(a.nwin, 1) ** 0.5
    print(f"\n== KEPUTUSAN E7b ==\n  kurva-U {uL-uS:+.4f} · SE={se:.4f} (2xSE={2*se:.4f})", flush=True)
    with open(a.jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(ntrain=a.ntrain, table_bits=a.table_bits, nwin=a.nwin,
                                win=a.win, snap_at=snap_at, se=se, results=res,
                                sd=sd)) + "\n")
    print(f"ditulis -> {a.jsonl}")


if __name__ == "__main__":
    main()
