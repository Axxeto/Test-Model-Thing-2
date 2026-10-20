"""Analisis ulang hasil E1 (cangkok) dengan verdict STATISTIS.

Kenapa ada: verdict di dalam `e1_graft.py` memakai versi skrip yang berjalan saat itu. Bila
skrip sudah diperbarui setelah proses dimulai, verdict yang tercetak mengikuti versi lama.
Skrip ini membaca `results` mentah dari `results/e1_graft.jsonl` lalu menerapkan aturan
KEJUJURAN yang sama: sebuah Δ cangkok hanya disimpulkan bila melampaui 2×SE antarsjendela.

Jalan:
  python scripts/analyze_e1.py
  python scripts/analyze_e1.py --jsonl results/e1_graft.jsonl --tail 1
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def verdict(delta, se_window, nwin, mode):
    """mode: 'pulih' (H-B bila Δ<0) atau 'rusak' (H-B bila Δ>0)."""
    thr = 2.0 * se_window / math.sqrt(max(nwin, 1))
    if not math.isfinite(se_window) or abs(delta) < thr:
        return f"TAK DAPAT DISIMPULKAN (|Δ|={abs(delta):.4f} < 2×SE={thr:.4f})"
    if mode == "pulih":
        return ("MEMULIHKAN -> MIXER/APM penyebab (H-B)" if delta < 0
                else "tidak memulihkan -> arah H-A tabel")
    return ("MERUSAK -> menguatkan MIXER/APM (H-B)" if delta > 0
            else "tidak merusak -> bukan mixer/APM saja")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", default="results/e1_graft.jsonl")
    ap.add_argument("--tail", type=int, default=0,
                    help="ambil N entri terakhir (0 = semua)")
    a = ap.parse_args()

    if not os.path.exists(a.jsonl):
        raise SystemExit(f"belum ada {a.jsonl} — jalankan e1_graft.py dulu")

    rows = []
    with open(a.jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if a.tail:
        rows = rows[-a.tail:]

    for r in rows:
        res = r["results"]
        nwin = r.get("nwin", 20)
        keys = sorted(res.keys(), key=lambda k: (0 if k.startswith("utuh") else 1, k))
        print(f"\n=== {r.get('corpus')} tb={r.get('table_bits')} ntrain={r.get('ntrain')} "
              f"nwin={nwin} win={r.get('win')} ===")
        for k in keys:
            print(f"  {k:34s}: {res[k]:.4f} bit")

        utuh = [k for k in keys if k.startswith("utuh")]
        if len(utuh) < 2:
            continue
        # urutkan menurut titik byte (numerik), bukan alfabet ("18jt" < "3jt" secara leksikal)
        def _jt(k):
            import re
            mnum = re.search(r"@(\d+)jt", k)
            return int(mnum.group(1)) if mnum else 0
        utuh = sorted(utuh, key=_jt)
        k0, k1 = utuh[0], utuh[-1]
        m0, m1 = _jt(k0), _jt(k1)
        tot = res[k1] - res[k0]
        # SE antarsjendela: pakai yang tersimpan bila ada (eksak); jika tidak, pakai
        # `se_window`; jika itu pun tak ada, asumsi konservatif 0,05 bit.
        se_map = r.get("noisse") or {}
        se = se_map.get(k1, r.get("se_window", 0.05))
        print(f"\n  utuh {k0} -> {k1}: {tot:+.4f} bit")
        print(f"  SE antarsjendela({k1})={se:.4f} -> 2×SE/√nwin = {2*se/math.sqrt(nwin):.4f}")
        kr = f"tab@{m1}jt+mix@{m0}jt"
        kd = f"tab@{m0}jt+mix@{m1}jt"
        for kk, mode in ((kr, "pulih"), (kd, "rusak")):
            if kk in res:
                base = res[k1] if mode == "pulih" else res[k0]
                print(f"  {kk:34s}: Δ={res[kk]-base:+.4f} -> {verdict(res[kk]-base, se, nwin, mode)}")


if __name__ == "__main__":
    main()
