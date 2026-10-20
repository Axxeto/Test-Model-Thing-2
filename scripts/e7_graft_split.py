"""E7 — CANGKOK TERPISAH: APM vs MIXER (lanjutan E6).

Latar:
  E1 (HASIL_E1_E2_KURVA_U.md): `tab@30jt + mixer/APM@3jt` MEMULIHKAN 100% -> pembawa kerusakan
    ada di dalam {w1, w2, apm1, apm2}.
  E6 (HASIL_E6_WBOUND.md): meng-clip bobot mixer lapis-1 (w_bound 8/16/32) TIDAK memulihkan apa pun
    -> magnitudo w1 BUKAN penyebabnya (peledakan itu gejala, bukan sebab).

E7 memisahkan sisa tersangka dengan mencangkok SUB-KOMPONEN secara terpisah pada protokol bersih
(suite TETAP 20x200 kB i.i.d. dari region uji; satuan CE = BIT; jendela tidak menyentuh latih):

  A. utuh@3jt / utuh@30jt          -> pembanding level
  B. tab@30 + {w1,w2}@3 + apm@30   -> bila PULIH ~ level 3jt  => pelakunya MIXER (w1/w2)
  C. tab@30 + apm@3 + {w1,w2}@30   -> bila PULIH ~ level 3jt  => pelakunya APM
  D. tab@30 + semua@3              -> kontrol pemulihan penuh (harus ~level 3jt; ulang E1)
  E. tab@3  + semua@30             -> kontrol reproduksi kerusakan (harus ~level 30jt)

Jalan:
  python scripts/e7_graft_split.py --ntrain 30000000 --table_bits 25 \
      --snap_at 3000000,30000000 --jsonl results/e7_graft_split.jsonl
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from tmt2.stat_core import StatCore

ARGS = None


def snapshot(m):
    """Salin komponen state (bukan referensi)."""
    return dict(tab=m.tab.copy(), w1=m.w1.copy(), w2=m.w2.copy(),
                apm1=m.apm1.copy(), apm2=m.apm2.copy())


def build_core(a, tab=None, w12=None, apm=None):
    """Bangun StatCore PRODUKSI langsung (bukan lewat Config) + tanam sub-komponen opsional."""
    orders = tuple(int(x) for x in a.orders.split(","))
    m = StatCore(orders=orders, table_bits=a.table_bits, wsel_bits=a.wsel_bits,
                 cks_bits=a.cks_bits, match_fix=True, ctx_fix=True, use_apm=True)
    if tab is not None:
        m.tab[:] = tab["tab"]
    if w12 is not None:
        m.w1[:] = w12["w1"]
        m.w2[:] = w12["w2"]
    if apm is not None:
        m.apm1[:] = apm["apm1"]
        m.apm2[:] = apm["apm2"]
    return m


def eval_suite(a, tab, w12, apm, windows):
    """CE rata-rata pada SEMUA jendela suite; model baru (state direset) tiap jendela; beku."""
    per = []
    for w in windows:
        m = build_core(a, tab=tab, w12=w12, apm=apm)
        per.append(float(np.nanmean(m.eval_stream_fast(w))))
    return float(np.mean(per)), per


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/indonesian_big.bin")
    ap.add_argument("--ntrain", type=int, default=30_000_000)
    ap.add_argument("--table_bits", type=int, default=25)
    ap.add_argument("--wsel_bits", type=int, default=12)
    ap.add_argument("--cks_bits", type=int, default=24)
    ap.add_argument("--orders", default="1,2,3,4,5,6,8,12,16,24")
    ap.add_argument("--snap_at", default="3000000,30000000")
    ap.add_argument("--nwin", type=int, default=20)
    ap.add_argument("--win", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--jsonl", default="results/e7_graft_split.jsonl")
    a = ap.parse_args()
    ARGS = a

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

    print(f"[E7] tb={a.table_bits} latih {a.ntrain} byte · suite {a.nwin}x{a.win}B i.i.d. tetap",
          flush=True)
    m = build_core(a)
    snaps = {}
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
            snaps[prev] = snapshot(m)
            print(f"    -> snapshot @ {prev/1e6:.1f} jt diambil", flush=True)

    labels = list(snaps.keys())
    S, L = labels[0], labels[-1]          # kecil (3jt), besar (30jt)
    res, se = {}, {}
    print("\n[E7] evaluasi cangkok (semua pada suite TETAP yang sama):", flush=True)

    def run(label, tab, w12, apm):
        bpb, per = eval_suite(a, tab, w12, apm, windows)
        res[label] = bpb
        se[label] = float(np.std(per))
        print(f"  {label:34s}: {bpb:.4f} bit  (sd {np.std(per):.4f})", flush=True)
        return bpb

    uS = run(f"utuh@{S//1000000}jt", snaps[S], snaps[S], snaps[S])
    uL = run(f"utuh@{L//1000000}jt", snaps[L], snaps[L], snaps[L])
    # B: mixer dari kecil, APM dari besar
    b_mix = run(f"tab@{L//1000000}+mix@{S//1000000}+apm@{L//1000000}",
                snaps[L], snaps[S], snaps[L])
    # C: APM dari kecil, mixer dari besar
    c_apm = run(f"tab@{L//1000000}+mix@{L//1000000}+apm@{S//1000000}",
                snaps[L], snaps[L], snaps[S])
    # D: kontrol pemulihan penuh
    d_all = run(f"tab@{L//1000000}+all@{S//1000000}", snaps[L], snaps[S], snaps[S])
    # E: kontrol reproduksi kerusakan
    e_bad = run(f"tab@{S//1000000}+all@{L//1000000}", snaps[S], snaps[L], snaps[L])

    # ---- verdict (ambang 2xSE antarsjendela) ----
    nwin = a.nwin
    seL = se[f"utuh@{L//1000000}jt"] / max(nwin, 1) ** 0.5
    print("\n== KEPUTUSAN E7 ==", flush=True)
    print(f"  level kecil@{S//1000000}jt = {uS:.4f} · level besar@{L//1000000}jt = {uL:.4f} "
          f"(kurva-U {uL-uS:+.4f})", flush=True)
    print(f"  SE antarsjendela (utuh akhir) = {seL:.4f} bit (ambang 2xSE = {2*seL:.4f})", flush=True)
    verdict = {}
    for name, val in [("MIXER (w1/w2)", b_mix), ("APM", c_apm)]:
        d = val - uL
        pulih = d < -2 * seL
        frac = (uL - val) / max(uL - uS, 1e-9)
        verdict[name] = dict(bpb=val, delta_vs_utuh_akhir=d, pulih=bool(pulih),
                             frac_pulih=float(frac))
        tag = "PULIH" if pulih else "tidak (di dalam noise)"
        print(f"  {name:14s}: {val:.4f} (Δ {d:+.4f} vs utuh akhir) -> {tag}, "
              f"memulihkan {frac*100:.0f}% kurva-U", flush=True)
    print(f"  kontrol: all@{S//1000000}jt = {d_all:.4f} (harus ~{uS:.4f}) · "
          f"tab@{S//1000000}+all@{L//1000000} = {e_bad:.4f} (harus ~{uL:.4f})", flush=True)

    with open(a.jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(corpus=a.corpus, ntrain=a.ntrain, table_bits=a.table_bits,
                                nwin=a.nwin, win=a.win, snap_at=snap_at,
                                se_window=seL, results=res, sd=se, verdict=verdict)) + "\n")
    print(f"\nditulis -> {a.jsonl}")


if __name__ == "__main__":
    main()
