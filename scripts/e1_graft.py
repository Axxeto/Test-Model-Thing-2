"""E1 — CANGKOK CHECKPOINT: memutus hipotesis kurva-U (H-A tabel vs H-B mixer/APM).

Latar (lihat MASALAH_SAAT_INI.md / audit 2026-10-06 §3):
    Pada korpus 190 MB, BPB-uji turun lalu NAIK (kurva-U). Dua kandidat utama:
      H-A: mesin TABEL — penggantian buta + oversubskripsi (D-03), tabel jenuh.
      H-B: DRIFT MIXER/APM — LR tetap, tanpa regularisasi, masukan berkolinearitas tinggi.

Cara memutus (audit §3.4 E1, "cangkok"):
    Latih SATU model tb=25 pada ~30 jt byte sambil MENYIMPAN snapshot pada beberapa titik
    (3/9/18/30 jt). Karena format state memisahkan `tab` dari `w1/w2/apm1/apm2`, kita bisa
    menyilang-komponen:
      (a) tab@T + mixer/APM@T   (utuh pada waktu T)
      (b) tab@30 + mixer/APM@9  -> bila BPB PULIH ~ level 9 jt  => penyebab di MIXER/APM (H-B)
      (c) tab@9  + mixer/APM@30 -> bila BPB memburuk            => menguatkan H-B
    Evaluasi pada SUITE TETAP: 20 jendela 200 kB dokumen i.i.d. yang dikeluarkan dari latih
    (protokol bersih sesuai audit §3.1) + jendela-bor tail sebagai pembanding lama.

PENTING (aturan proyek): jendela uji WAJIB tetap & tidak menyentuh latih; satuan CE = BIT
(eval_stream sudah bit; JANGAN /LN2 — bug B-01).

Jalan:
  python scripts/e1_graft.py --ntrain 30000000 --table_bits 25
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from tmt2.model import Config
from tmt2.stat_core import StatCore


def snapshot(m):
    """Salin komponen state (bukan referensi) untuk cangkok nanti."""
    return dict(tab=m.tab.copy(), w1=m.w1.copy(), w2=m.w2.copy(),
                apm1=m.apm1.copy(), apm2=m.apm2.copy(),
                mchk=m.mchk.copy(), mbyte=m.mbyte.copy())


def build_core(a, snap=None):
    """Bangun StatCore PRODUKSI langsung (bukan lewat Config: lihat B-06 — `from_config`
    tidak meneruskan cks_bits/match_fix/ctx_fix/count_cap). Opsional tanam snapshot."""
    orders = tuple(int(x) for x in a.orders.split(","))
    m = StatCore(orders=orders, table_bits=a.table_bits, wsel_bits=a.wsel_bits,
                 cks_bits=a.cks_bits, match_fix=True, ctx_fix=True, use_apm=True)
    if snap is not None:
        if snap.get("tab") is not None:
            m.tab[:] = snap["tab"]
        if snap.get("w1") is not None:
            m.w1[:] = snap["w1"]
        if snap.get("w2") is not None:
            m.w2[:] = snap["w2"]
        if snap.get("apm1") is not None:
            m.apm1[:] = snap["apm1"]
        if snap.get("apm2") is not None:
            m.apm2[:] = snap["apm2"]
    return m


def eval_suite(tab_component, mixer_component, windows):
    """CE rata-rata pada SEMUA jendela suite, model baru tiap jendela (state direset).

    Model beku (train=False) — konsisten dengan protokol evaluasi proyek.
    Mengembalikan mean bit/byte dan list per-jendela.
    """
    per = []
    for w in windows:
        m = build_core(ARGS, snap={"tab": tab_component.get("tab"),
                                   "w1": mixer_component.get("w1"),
                                   "w2": mixer_component.get("w2"),
                                   "apm1": mixer_component.get("apm1"),
                                   "apm2": mixer_component.get("apm2")})
        ce = m.eval_stream_fast(w)
        per.append(float(np.nanmean(ce)))
    return float(np.mean(per)), per


NOISE_SE = {}


ARGS = None


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/indonesian_big.bin")
    ap.add_argument("--ntrain", type=int, default=30_000_000)
    ap.add_argument("--table_bits", type=int, default=25)
    ap.add_argument("--wsel_bits", type=int, default=12)
    ap.add_argument("--cks_bits", type=int, default=24)
    ap.add_argument("--orders", default="1,2,3,4,5,6,8,12,16,24")
    ap.add_argument("--snap_at", default="3000000,9000000,18000000,30000000",
                    help="titik byte untuk snapshot komponen")
    ap.add_argument("--nwin", type=int, default=20, help="jumlah jendela uji")
    ap.add_argument("--win", type=int, default=200_000, help="ukuran jendela uji (byte)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--jsonl", default="results/e1_graft.jsonl")
    a = ap.parse_args()
    ARGS = a

    snap_at = sorted(int(x) for x in a.snap_at.split(",") if x.strip())
    need = a.ntrain + a.nwin * a.win
    with open(a.corpus, "rb") as f:
        raw = np.frombuffer(f.read(need), dtype=np.uint8)
    if len(raw) < need:
        raise SystemExit(f"korpus kurang: {len(raw)} < {need}")
    tr = raw[:a.ntrain]
    # SUITE TETAP: nwin jendela i.i.d. dari region uji (di luar latih) — audit §3.1 protokol bersih
    test_region = raw[a.ntrain:]
    rng = np.random.default_rng(a.seed)
    starts = rng.choice(len(test_region) - a.win, size=a.nwin, replace=False)
    windows = [test_region[s:s + a.win] for s in starts]

    print(f"[E1] latih tb={a.table_bits} pada {a.ntrain} byte · suite {a.nwin}x{a.win}B i.i.d.",
          flush=True)
    m = build_core(a)

    snaps = {}
    t0 = time.perf_counter()
    # ⚡ Jalur CEPAT: latih per-INTERVAL dalam satu panggilan numba (train_stream), bukan
    # loop byte Python (dulu membatasi ~90 kB/s). Snapshot diambil di antara interval.
    bounds = sorted(set(snap_at) | {a.ntrain})
    prev = 0
    for nb in bounds:
        if nb > a.ntrain:
            nb = a.ntrain
        if nb > prev:
            m.train_stream(tr[prev:nb])
            prev = nb
        el = time.perf_counter() - t0
        print(f"  latih sampai {prev/1e6:.1f} jt byte "
              f"({el:.0f}s, {prev/max(el,1e-9):.0f} B/s)", flush=True)
        if prev in snap_at:
            snaps[prev] = snapshot(m)
            print(f"    -> snapshot @ {prev/1e6:.1f} jt diambil", flush=True)

    # ---- evaluasi cangkok ----
    labels = list(snaps.keys())
    print("\n[E1] evaluasi (semua pada suite TETAP yang sama):", flush=True)
    results = {}
    for T in labels:
        bpb, per = eval_suite(snaps[T], snaps[T], windows)
        results[f"utuh@{T//1000000}jt"] = bpb
        NOISE_SE[f"utuh@{T//1000000}jt"] = float(np.std(per))
        print(f"  utuh@{T/1e6:.0f}jt        : {bpb:.4f} bit  (sd {np.std(per):.4f})", flush=True)

    # cangkok silang: tabel dari titik besar, mixer/APM dari titik kecil dan sebaliknya
    last = labels[-1]
    if len(labels) >= 2:
        for T in labels[:-1]:
            bpb, per = eval_suite(snaps[last], snaps[T], windows)
            results[f"tab@{last//1000000}jt+mix@{T//1000000}jt"] = bpb
            print(f"  tab@{last/1e6:.0f}jt+mix@{T/1e6:.0f}jt: {bpb:.4f} bit  (sd {np.std(per):.4f})",
                  flush=True)
            bpb, per = eval_suite(snaps[T], snaps[last], windows)
            results[f"tab@{T//1000000}jt+mix@{last//1000000}jt"] = bpb
            print(f"  tab@{T/1e6:.0f}jt+mix@{last/1e6:.0f}jt: {bpb:.4f} bit  (sd {np.std(per):.4f})",
                  flush=True)

    # ---- keputusan (pakai snapshot pertama & terakhir, apa pun titiknya) ----
    # ⚠️ KEJUJURAN: ambang 0,5×|tot| saja bisa digerakkan NOISE. Bila |Δ cangkok| < 2×SE
    # (SE = sd antarsjendela / sqrt(nwin)), verdict TIDAK dapat disimpulkan -> nyatakan begitu
    # alih-alih memaksa jawaban. Ini penting karena amplitudo kurva-U bisa ~0,03 bit sementara
    # sd antarsjendela ~0,05 bit.
    print("\n== KEPUTUSAN E1 ==")
    T0, T1 = labels[0], labels[-1]
    u0 = results[f"utuh@{T0//1000000}jt"]
    u1 = results[f"utuh@{T1//1000000}jt"]
    tot = u1 - u0
    nwin = a.nwin
    # SE sinyal per-jendela terbaik yang tersedia: dari utuh (paling relevan)
    se = NOISE_SE.get(f"utuh@{T1//1000000}jt", float("nan"))
    print(f"  utuh {T0/1e6:.0f}jt -> {T1/1e6:.0f}jt      : {tot:+.4f} bit (+ = naik/kurva-U)")
    print(f"  SE antarsjendela (utuh akhir) : {se:.4f} bit / sqrt({nwin}) = {se/max(nwin,1)**0.5:.4f}")

    def _verdict(delta, base, direksi_duga):
        """False bila |delta| tak melampaui 2×SE -> tak dapat disimpulkan."""
        if not np.isfinite(se) or abs(delta) < 2.0 * se / max(nwin, 1) ** 0.5:
            return "TAK DAPAT DISIMPULKAN (Δ di dalam noise, < 2×SE)"
        if direksi_duga == "pulih":
            return ("MEMULIHKAN ke level awal -> MIXER/APM penyebab (H-B)"
                    if delta < 0 else "tidak memulihkan -> arah H-A tabel")
        return ("MERUSAK level awal -> menguatkan MIXER/APM (H-B)"
                if delta > 0 else "tidak merusak -> bukan mixer/APM saja")

    k_rec = f"tab@{T1//1000000}jt+mix@{T0//1000000}jt"
    k_deg = f"tab@{T0//1000000}jt+mix@{T1//1000000}jt"
    if k_rec in results:
        rec = results[k_rec] - u1
        print(f"  cangkok tab@{T1/1e6:.0f}jt + mix@{T0/1e6:.0f}jt : {results[k_rec]:.4f} bit "
              f"(Δ {rec:+.4f} vs utuh akhir) -> {_verdict(rec, u1, 'pulih')}")
    if k_deg in results:
        deg = results[k_deg] - u0
        print(f"  cangkok tab@{T0/1e6:.0f}jt + mix@{T1/1e6:.0f}jt : {results[k_deg]:.4f} bit "
              f"(Δ {deg:+.4f} vs utuh awal) -> {_verdict(deg, u0, 'rusak')}")
    if abs(tot) < 2.0 * se / max(nwin, 1) ** 0.5:
        print("  ⚠️ PERINGATAN: goyangan utuh awal->akhir SENDIRI ada di dalam noise -> "
              "kurva-U tak terbukti pada titik ini; jangan simpulkan penyebab.")

    with open(a.jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(corpus=a.corpus, ntrain=a.ntrain, table_bits=a.table_bits,
                                nwin=a.nwin, win=a.win, snap_at=snap_at,
                                se_window=se, noisse=NOISE_SE,
                                results=results)) + "\n")
    print(f"\nditulis -> {a.jsonl}")


if __name__ == "__main__":
    main()
