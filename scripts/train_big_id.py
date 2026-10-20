"""Latih Tingkat A pada korpus Indonesia BESAR (hasil ekstraksi parquet) + hasilkan teks.

Jalur terbaik: Tingkat A (inti statistik biner). Konfigurasi PRODUKSI:
  table_bits=25, wsel_bits=12, APM on, match on, word on, sparse OFF.

Perbedaan dari train_gen_id.py:
  * Korpus ratusan MB (bukan 1 MB) -> aliran dipecah per-jendela agar tidak memuat numpy
    array raksasa berulang; CE dihitung pada JENDELA UJI TETAP = EKOR aliran.
  * Tulis titik-kontrol progres tiap ``--report_every`` byte supaya laju bisa dipantau
    untuk pelatihan berjam-jam; JSONL di-flush tiap titik.
  * Dukung ``--ntrain`` untuk memotong (uji cepat) tanpa mengubah jendela uji.

PROTOKOL (wajib, dari DOKUMENTASI_LENGKAP.md):
  * Prekuensial: byte ke-t diprediksi SEBELUM dilihat.
  * Jendela uji = ekor aliran, SAMA untuk semua ukuran latih (jebakan #4).
  * Metrik tunggal = BPB (bit/byte).

Jalan:
  python scripts/train_big_id.py                        # seluruh korpus
  python scripts/train_big_id.py --ntrain 5000000       # uji cepat 5 MB
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

PROMPTS = [
    "Indonesia adalah negara kepulauan yang",
    "Pada tahun 1945, para pendiri bangsa",
    "Pemerintah Indonesia terus berupaya meningkatkan",
    "Menurut para ahli, perubahan iklim",
    "Jakarta merupakan ibu kota negara",
    "Salah satu makanan khas Indonesia adalah",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--ntrain", type=int, default=0, help="0 = seluruh korpus minus uji")
    ap.add_argument("--ntest", type=int, default=100000, help="jendela uji = ekor aliran")
    ap.add_argument("--table_bits", type=int, default=25)
    ap.add_argument("--wsel_bits", type=int, default=12)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--orders", default="1,2,3,4,5,6,8,12,16,24")
    ap.add_argument("--no_apm", action="store_true")
    ap.add_argument("--sparse", action="store_true", help="sparse = MERUGI; default mati")
    ap.add_argument("--gen", type=int, default=800)
    ap.add_argument("--temp", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warm", type=int, default=4000)
    ap.add_argument("--report_every", type=int, default=2000000,
                    help="tulis progres tiap n byte")
    ap.add_argument("--probe_test", type=int, default=20000,
                    help="ukur BPB pada awal jendela uji tiap laporan (0 = mati; ini MAHAL)")
    ap.add_argument("--out", default="results/gen_big_id.txt")
    a = ap.parse_args()

    orders = tuple(int(x) for x in a.orders.split(","))

    # --- muat aliran via mmap-like read lalu potong (hindari salinan ganda)
    t_load = time.perf_counter()
    with open(a.corpus, "rb") as f:
        raw = np.frombuffer(f.read(), dtype=np.uint8)
    total = raw.size
    # ⚠️ PROTOKOL UJI. Dua jendela uji yang sah, jangan dicampur:
    #   * default (`--ntrain 0`): latih = prefix [0, total-ntest), uji = ekor [total-ntest, total).
    #     Tak beririsan -> bersih.
    #   * `--ntrain N` (N < total-ntest): latih = prefix [0,N), uji TETAP ekor. Tak beririsan,
    #     tetapi jendela uji jauh dari data latih (topik bisa berbeda) -> catat saat menafsirkan.
    # DILARANG memakai jendela uji yang beririsan dengan prefix latih: itu kebocoran (jebakan
    # kebocoran KEEMPAT; terukur BPB palsu 0,53 vs 2,14 yang jujur).
    ntrain = a.ntrain if a.ntrain > 0 else total - a.ntest
    assert ntrain + a.ntest <= total, (
        f"jendela uji ekor [{total-a.ntest},{total}) beririsan dengan latih [0,{ntrain}) "
        f"-> kebocoran; kurangi --ntrain atau --ntest")
    te = raw[total - a.ntest:]
    tr = raw[:ntrain]
    gap = (total - a.ntest) - ntrain
    print(f"[big] korpus {total} B · latih {ntrain} B · uji {a.ntest} B (ekor) · "
          f"jarak latih->uji {gap} B · muat {time.perf_counter() - t_load:.1f} s", flush=True)
    print(f"[big] Tingkat A · table_bits={a.table_bits} · wsel_bits={a.wsel_bits} · "
          f"APM={'off' if a.no_apm else 'on'} · sparse={'on' if a.sparse else 'off'} · "
          f"orders={orders}", flush=True)

    m = StatCore(orders=orders, table_bits=a.table_bits, lr=a.lr,
                 use_apm=not a.no_apm, wsel_bits=a.wsel_bits,
                 sparse=a.sparse, word=True)

    # --- latih dengan pelaporan berkala.
    # CE dilaporkan sebagai rata-rata KUMULATIF **dan** KUMULATIF-SEGMEN (potongan
    # ``report_every`` terakhir). Yang kedua itulah indikator jujur: rata-rata kumulatif
    # praktis tidak bergerak setelah puluhan juta byte, sehingga lonjakan pada segmen
    # terakhir (data lebih sukar) tampak seperti "model mundur". Juga kita ukur BPB pada
    # jendela uji tetap secara berkala (mahal, O(ntest)) untuk melihat kurva uji sebenarnya.
    t0 = time.perf_counter()
    warm = min(a.warm, len(tr) - 1)
    for b in tr[:warm]:
        m.step(b)
    t_warm = time.perf_counter()

    next_report = a.report_every
    seg_start = warm                      # awal potongan sekarang
    seg_bits = 0.0                        # CE-bit terakumulasi dalam potongan
    seg_cnt = 0
    n = warm
    for b in tr[warm:]:
        c = m.step(b)
        n += 1
        if c is not None:
            seg_bits += c
            seg_cnt += 1
        if n >= next_report:
            el = time.perf_counter() - t_warm
            bps = (n - warm) / max(el, 1e-9)
            eta = (len(tr) - n) / max(bps, 1e-9)
            seg_ce = seg_bits / max(seg_cnt, 1)
            cum_ce = m.ce_bits / max(m.n_bytes, 1)
            # BPB pada jendela uji (bisa dimatikan dengan --probe_test 0)
            probe = ""
            if a.probe_test:
                pt = te[:min(a.probe_test, len(te))]
                ce_t = float(m.eval_stream(pt).mean())
                probe = f" · BPB-uji(awal {len(pt)}B) {ce_t:.4f}"
            print(f"[big] {n:>12,} B · {bps:>7.0f} B/s · "
                  f"CE-seg {seg_ce:.4f} · CE-cum {cum_ce:.4f}{probe} · ETA {eta/60:.1f} m",
                  flush=True)
            next_report += a.report_every
            seg_start = n
            seg_bits = 0.0
            seg_cnt = 0

    dt_train = time.perf_counter() - t0
    bps = len(tr) / max(dt_train, 1e-9)
    print(f"[big] latih selesai {dt_train/60:.2f} m · {bps:.0f} B/s "
          f"(rata-rata seluruh latih)", flush=True)

    # --- diagnostik kesehatan tabel: berapa slot terisi + tingkat tabrakan.
    # Kalau hampir semua slot terisi, checksum 8-bit mulai bertabrakan → kualitas
    # berhenti naik walau data ditambah. Ini penjelas "plafon" yang lebih penting
    # daripada CE-latih sesaat.
    used = int(np.count_nonzero(m.tab))
    size = m.tab.size
    print(f"[big] tabel: {used:,}/{size:,} slot terisi ({100.0*used/size:.1f}%)", flush=True)

    # --- evaluasi (jendela uji tetap)
    t_e = time.perf_counter()
    ce = float(m.eval_stream(te).mean())
    print(f"[big] BPB tahan (uji {a.ntest} B): {ce:.4f} bit/byte · "
          f"eval {time.perf_counter() - t_e:.1f} s", flush=True)

    # --- generasi
    lines = [f"# Generasi Tingkat A — korpus Indonesia BESAR ({total} B latih {ntrain} B)",
             f"# BPB tahan {ce:.4f} · temp {a.temp} · table_bits {a.table_bits} · wsel {a.wsel_bits}",
             ""]
    for i, p in enumerate(PROMPTS):
        for s in range(2):
            txt = m.generate(p, a.gen, temp=a.temp, seed=a.seed + i * 10 + s)
            dec = txt.decode("utf-8", errors="replace")
            lines.append(f"=== prompt: {p!r}  (seed {a.seed + i * 10 + s}) ===")
            lines.append(p + dec)
            lines.append("")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[big] generasi ditulis -> {a.out}", flush=True)

    os.makedirs("results", exist_ok=True)
    with open("results/gen_big_id.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(corpus=a.corpus, total_bytes=int(total), ntrain=int(ntrain),
                                ntest=a.ntest, bpb=ce, bps=bps, train_s=dt_train,
                                table_bits=a.table_bits, wsel_bits=a.wsel_bits,
                                table_fill=round(used / size, 4),
                                sparse=a.sparse, temp=a.temp, gen=a.gen, seed=a.seed)) + "\n")

    print("\n---- cuplikan generasi ----", flush=True)
    for p in PROMPTS[:3]:
        txt = m.generate(p, 320, temp=a.temp, seed=a.seed)
        print(f"\n[{p}]")
        print((p + txt.decode("utf-8", errors="replace"))[:420], flush=True)


if __name__ == "__main__":
    main()
