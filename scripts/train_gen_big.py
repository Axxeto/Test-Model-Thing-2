"""Latih Tingkat A pada korpus Indonesia BESAR (30 juta byte) lalu hasilkan teks.

Konfigurasi produksi (hasil investigasi kurva-U 2026-10-05):
  * table_bits=26  — LEBIH BAIK dari 25: pada 14 jt byte CE 1,978 (masih turun) vs
    25: 2,048 (sudah berbalik naik). Tabel 2× lebih besar = tekanan penimpaan separuh.
  * DUA PERBAIKAN KODE aktif (default): checksum match dari bit-atas (bukan bit indeks),
    dan indeks konteks `c` ikut di-hash. Gabungan −0,0616 bit.
  * sparse OFF (terbukti merugi), APM ON (terbukti menolong, +0,058), word ON, wsel=12.
  * cks_bits=24 (checksum konteks lebar → tabrakan lebih jarang).

Skrip ini: (1) latih 30 jt byte, (2) laporkan BPB pada jendela uji tetap 200 kB SETELAH
prefix latih (tak beririsan), (3) simpan checkpoint .npz, (4) hasilkan sampel dari
beberapa prompt × suhu, (5) tulis laporan teks + JSONL.

Jalan:
  python scripts/train_gen_big.py --ntrain 30000000 --ntest 200000
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

CORPUS = "data/indonesian_big.bin"

PROMPTS = [
    "Indonesia adalah negara kepulauan yang",
    "Pada tahun 1945, para pendiri bangsa",
    "Pemerintah mengumumkan kebijakan baru untuk",
    "Kucing merupakan hewan mamalia yang",
    "Teknologi informasi telah mengubah cara",
]


def save_ckpt(path, m, meta):
    # Delegasi ke StatCore.save (sumber kebenaran tunggal format ckpt, §11.3).
    m.save(path, meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--ntrain", type=int, default=30_000_000)
    ap.add_argument("--ntest", type=int, default=200_000)
    ap.add_argument("--table_bits", type=int, default=26)
    ap.add_argument("--wsel_bits", type=int, default=12)
    ap.add_argument("--lr", type=float, default=0.002)
    ap.add_argument("--cks_bits", type=int, default=24)
    ap.add_argument("--orders", default="1,2,3,4,5,6,8,12,16,24")
    ap.add_argument("--no_apm", action="store_true")
    ap.add_argument("--sparse", action="store_true", help="ablasi: konteks jarang (merugi)")
    ap.add_argument("--gen", type=int, default=400, help="byte per sampel")
    ap.add_argument("--report_every", type=int, default=5_000_000)
    ap.add_argument("--freeze", type=int, default=0,
                    help="mixer_freeze_after: bekukan mixer+APM setelah N byte (0=tak pernah; fix kurva-U E8)")
    ap.add_argument("--w_bound", type=float, default=0.0,
                    help="batas |bobot| mixer lapis-1 (0=tanpa batas; E6: bukan solusi, pengaman saja)")
    ap.add_argument("--mixer_mode", type=int, default=1,
                    help="0=legacy (angka G1 lama) | 1=Mixer v3 galat-sendiri (default sejak gerbang §7)")
    ap.add_argument("--sched", type=int, default=0, metavar="NHALF",
                    help="jadwal per-baris: kunjungan setengah-waktu (0=off; v3: 50000)")
    ap.add_argument("--sched_floor", type=float, default=0.02,
                    help="lantai faktor jadwal (WAJIB <1 bila --sched>0)")
    ap.add_argument("--mix_l2", type=float, default=0.0, help="tarikan L2 v3 (default off)")
    ap.add_argument("--drop", type=int, default=0, help="dropout konteks permille (0=mati)")
    ap.add_argument("--match_adapt", type=int, default=0, help="1=keyakinan match adaptif (C2)")
    ap.add_argument("--hitsel", type=int, default=1, help="1=selector set-0 per hit-order (default)")
    ap.add_argument("--match_bits", type=int, default=24, help="bit tabel match (24 default)")
    ap.add_argument("--s2_clip", type=float, default=16.0, help="batas input lapis-2 v3")
    ap.add_argument("--ckpt", default="checkpoints/id_big_30m.npz")
    ap.add_argument("--out", default="results/gen_big_id_30m.txt")
    ap.add_argument("--jsonl", default="results/train_gen_big.jsonl")
    a = ap.parse_args()

    orders = tuple(int(x) for x in a.orders.split(","))
    with open(a.corpus, "rb") as f:
        raw = np.frombuffer(f.read(a.ntrain + a.ntest), dtype=np.uint8)
    total = raw.size
    if a.ntrain + a.ntest > total:
        raise SystemExit(f"korpus hanya {total} B, butuh {a.ntrain + a.ntest}")
    tr = raw[:a.ntrain]
    te = raw[a.ntrain:a.ntrain + a.ntest]      # jendela uji TETAP, tak beririsan dgn latih

    print(f"KORPUS Indonesia besar · {a.corpus} · {total:,} B dibaca")
    print(f"latih {a.ntrain:,} B · uji {a.ntest:,} B (jendela tetap SETELAH prefix latih)")
    print(f"Tingkat A · tb={a.table_bits} · wsel={a.wsel_bits} · cks={a.cks_bits} · "
          f"APM={'off' if a.no_apm else 'on'} · sparse={a.sparse} · fix=ON")
    print(f"orders={orders}\n", flush=True)

    m = StatCore(orders=orders, table_bits=a.table_bits, lr=a.lr,
                 use_apm=not a.no_apm, wsel_bits=a.wsel_bits, sparse=a.sparse,
                 word=True, cks_bits=a.cks_bits, w_bound=a.w_bound,
                 mixer_freeze_after=a.freeze, mixer_mode=a.mixer_mode,
                 s2_clip=a.s2_clip, mix_lr_floor=a.sched_floor,
                 mix_lr_nhalf=a.sched, mix_l2=a.mix_l2,
                 dropout_permille=a.drop, match_adapt=a.match_adapt,
                 match_bits=a.match_bits, hit_sel=a.hitsel)

    # Jalur cepat terkompilasi per interval (§11.2): bit-identik dengan loop step(),
    # ~1,5-2x lebih cepat, dan menghormati freeze (loop step() lama TIDAK menghormatinya
    # sebelum fix _eff_rates — hasil lama tanpa --freeze tak terpengaruh karena freeze=0).
    t0 = time.perf_counter()
    n = 0
    next_rep = a.report_every
    while n < a.ntrain:
        nb = min(next_rep, a.ntrain) - n
        m.train_stream(tr[n:n + nb])
        n += nb
        if n >= next_rep:
            fill = float(np.count_nonzero(m.tab) / m.tab.size)
            print(f"  {n/1e6:5.0f} jt B · {n/max(time.perf_counter()-t0,1e-9):.0f} B/s · "
                  f"isi tabel {fill*100:.1f}% · max|w1| {float(np.abs(m.w1).max()):.1f}",
                  flush=True)
            next_rep += a.report_every
    tt = time.perf_counter() - t0
    bps = a.ntrain / max(tt, 1e-9)
    fill = float(np.count_nonzero(m.tab) / m.tab.size)
    print(f"\nlatih selesai: {tt:.1f} s · ~{bps:.0f} B/s · isi tabel {fill*100:.1f}%")

    # simpan keadaan konteks PASCA-latih supaya generasi tidak tercemar byte uji
    ctx_snap = (int(m.pos), int(m.hlen), int(m.prev_byte), m.hist.copy())

    ce = float(m.eval_stream(te).mean())
    print(f"BPB tahan (jendela uji {a.ntest:,} B): {ce:.4f}")

    # ---- pulihkan konteks pasca-latih DULU (buang jejak byte uji), BARU checkpoint:
    # ckpt harus menyimpan konteks akhir-latih, bukan akhir-uji — kalau tidak,
    # lanjut-latih dari ckpt terkontaminasi histori uji + reload tak bit-identik.
    m.pos, m.hlen, m.prev_byte, h = ctx_snap
    m.hist[:] = h

    # ---- checkpoint
    os.makedirs(os.path.dirname(a.ckpt), exist_ok=True)
    # ⚠️ B-06: simpan SELURUH hiperparameter via m.hyper() agar muat-ulang bit-identik.
    meta = dict(corpus=a.corpus, ntrain=a.ntrain, ntest=a.ntest, bpb=ce,
                table_bits=a.table_bits, wsel_bits=a.wsel_bits, cks_bits=a.cks_bits,
                orders=list(orders), use_apm=not a.no_apm, sparse=a.sparse, lr=a.lr,
                freeze=a.freeze, w_bound=a.w_bound, mixer_mode=a.mixer_mode,
                sched=a.sched, mix_l2=a.mix_l2, s2_clip=a.s2_clip,
                hyper=m.hyper())
    save_ckpt(a.ckpt, m, meta)
    print(f"checkpoint -> {a.ckpt}")

    # ---- generasi (konteks sudah pasca-latih murni)
    lines = [f"# Generasi Tingkat A — korpus Indonesia besar (latih {a.ntrain:,} B)",
             f"# BPB tahan {ce:.4f} · tb={a.table_bits} · wsel={a.wsel_bits} · cks={a.cks_bits}",
             f"# APM={'off' if a.no_apm else 'on'} · sparse={a.sparse} · fix=ON", ""]
    samples = []
    for i, p in enumerate(PROMPTS):
        for temp in (0.5, 0.75, 1.0):
            txt = m.generate(p, a.gen, temp=temp, seed=1234 + i * 10 + int(temp * 100))
            dec = txt.decode("utf-8", errors="replace")
            samples.append(dict(prompt=p, temp=temp, text=p + dec))
            lines.append(f"=== prompt: {p!r}  (temp {temp}) ===")
            lines.append(p + dec)
            lines.append("")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"generasi -> {a.out}")

    with open(a.jsonl, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(ntrain=a.ntrain, ntest=a.ntest, bpb=ce, bps=bps,
                                table_bits=a.table_bits, wsel_bits=a.wsel_bits,
                                cks_bits=a.cks_bits, use_apm=not a.no_apm,
                                sparse=a.sparse, fill=fill, train_s=tt)) + "\n")

    print("\n---- cuplikan (prompt #1, temp 0.75) ----")
    for s in samples:
        if s["prompt"] == PROMPTS[0] and s["temp"] == 0.75:
            print(s["text"][:500])
            break


if __name__ == "__main__":
    main()
