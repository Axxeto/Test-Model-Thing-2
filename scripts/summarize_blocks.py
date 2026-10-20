"""Ringkas hasil Eksperimen 1 (diag_blocks.jsonl) -> tabel + cek degradasi.

Pertanyaan: apakah rasio CE_kita / CE_bz2 NAIK sepanjang korpus?
  - rasio ~konstan  -> model melacak kesulitan data (tak ada degradasi internal)
  - rasio naik      -> degradasi internal model (tabel/match/penghitung)

Jalan: python scripts/summarize_blocks.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, "results", "diag_blocks.jsonl")


def load(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    rows.sort(key=lambda r: r["n0"])
    return rows


def show(rows, label):
    if not rows:
        print(f"{label}: kosong")
        return
    print(f"\n### {label}")
    print(f"{'blok':>4} {'MB0':>6} {'MB1':>6} {'CE':>8} {'bz2':>8} {'rasio':>6} {'isi%':>6}")
    for r in rows:
        print(f"{r['blk']:>4} {r['n0']/1e6:>6.0f} {r['n1']/1e6:>6.0f} "
              f"{r['ce_ours']:>8.4f} {r['ce_bz2']:>8.4f} {r['ratio']:>6.3f} "
              f"{r['fill']*100:>6.1f}")

    q = max(1, len(rows) // 4)
    first, last = rows[:q], rows[-q:]
    ce1 = sum(r["ce_ours"] for r in first) / len(first)
    ce2 = sum(r["ce_ours"] for r in last) / len(last)
    bz1 = sum(r["ce_bz2"] for r in first) / len(first)
    bz2 = sum(r["ce_bz2"] for r in last) / len(last)
    r1 = sum(r["ratio"] for r in first) / len(first)
    r2 = sum(r["ratio"] for r in last) / len(last)
    print(f"tren kuartal: CE {ce1:.4f}->{ce2:.4f} (d{ce2-ce1:+.4f}) · "
          f"bz2 {bz1:.4f}->{bz2:.4f} (d{bz2-bz1:+.4f}) · "
          f"rasio {r1:.3f}->{r2:.3f} (d{r2-r1:+.3f})")
    print("  " + ("rasio NAIK -> degradasi INTERNAL model." if r2 - r1 > 0.05
                  else "rasio ~konstan/turun: model melacak kesulitan data."))


def main():
    import sys as _sys
    if len(_sys.argv) > 1:
        for p in _sys.argv[1:]:
            if not os.path.exists(p):
                print(f"tidak ada {p}")
                continue
            show(load(p), os.path.basename(p))
        return
    if not os.path.exists(P):
        print(f"belum ada {P}")
        return
    show(load(P), os.path.basename(P))


if __name__ == "__main__":
    main()

