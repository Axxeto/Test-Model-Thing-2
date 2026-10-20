"""Instrumentasi MATCH: |w_match| (bobot mixer pada masukan match) & akurasi match.

Menjawab permintaan analisis: ukur seberapa besar pengaruh masukan match, dan
seberapa sering prediksi match benar. Ini penting karena (a) cacat checksum match
sudah diperbaiki, (b) perlu tahu apakah match masih "berbohong" setelah perbaikan.

DUA SUMBER:
  1. |w_match| = rata-rata |w1[m, sel, nctx]| untuk ketiga mixer. Disimpan LANGSUNG
     oleh kernel, tanpa perubahan kode. Kalau ~0, mixer MENGABAIKAN match; kalau
     besar, match berpengaruh kuat.
  2. Akurasi match: dihitung ulang di luar kernel dengan logika match yang SAMA
     (FNV atas `match_minlen` byte terakhir -> _mix64 -> slot + checksum). Kita
     bandingkan prediksi byte di slot match dengan byte sebenarnya, dan pisahkan
     menurut checksum MENCOCOK vs TIDAK (untuk memperlihatkan cacat vs perbaikan).

Jalan: python scripts/diag_match.py --ntrain 5000000 --table_bits 25
"""
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore", message="overflow encountered")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from tmt2.stat_core import StatCore, _mix64, HBUF

CORPUS = "data/indonesian_big.bin"
ORDERS = (1, 2, 3, 4, 5, 6, 8, 12, 16, 24)
FNV_OFF = np.uint64(0xCBF29CE484222325)
FNV_PRM = np.uint64(0x100000001B3)


def match_stats(raw, ntrain, table_bits, minlen, match_bits, match_fix, ctx_fix, cks_bits):
    m = StatCore(orders=ORDERS, table_bits=table_bits, lr=0.002, use_apm=True,
                 wsel_bits=12, sparse=False, word=True, cks_bits=cks_bits,
                 match_minlen=minlen, match_bits=match_bits,
                 match_fix=match_fix, ctx_fix=ctx_fix)
    mmask = m.mmask
    mck_mask = m.mck_mask
    data = raw[:ntrain]
    n = 0
    # statistik
    hit_ck = 0        # slot terisi & checksum cocok
    pred_ok = 0       # prediksi byte benar
    miss = 0          # slot belum pernah terisi
    for i in range(len(data)):
        if m.hlen >= minlen:
            hm = FNV_OFF
            for j in range(minlen):
                hm = (hm ^ np.uint64(m.hist[(m.pos - 1 - j) & (HBUF - 1)])) * FNV_PRM
            mkey = _mix64(hm)
            msl = int(mkey & np.uint64(mmask))
            if match_fix == 1:
                mck = int((mkey >> np.uint64(40)) & np.uint64(mck_mask))
            else:
                mck = int(mkey & np.uint64(255))
            if int(m.mchk[msl]) == mck and (m.mchk[msl] != 0 or m.mbyte[msl] != 0):
                hit_ck += 1
                if int(m.mbyte[msl]) == int(data[i]):
                    pred_ok += 1
            else:
                miss += 1
        m.step(int(data[i]))
        n += 1

    # |w_match| dari bobot mixer (indeks nctx = masukan match)
    nctx = m.nctx
    wm = np.abs(m.w1[:, :, nctx])
    wm_mean = float(wm.mean())
    # w juga untuk masukan tabel & neural sbg pembanding
    wtab = float(np.abs(m.w1[:, :, :nctx]).mean())
    wnn = float(np.abs(m.w1[:, :, nctx + 1]).mean())
    tot = hit_ck + miss
    return dict(ntrain=n, match_fix=match_fix, ctx_fix=ctx_fix,
                hit_ck=hit_ck, miss=miss,
                akurasi_match=pred_ok / max(hit_ck, 1),
                cakupan=hit_ck / max(tot, 1),
                w_match=wm_mean, w_tabel=wtab, w_neural=wnn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--ntrain", type=int, default=5_000_000)
    ap.add_argument("--table_bits", type=int, default=25)
    ap.add_argument("--match_bits", type=int, default=20)
    ap.add_argument("--match_minlen", type=int, default=6)
    ap.add_argument("--cks_bits", type=int, default=24)
    a = ap.parse_args()

    with open(a.corpus, "rb") as f:
        raw = np.frombuffer(f.read(a.ntrain), dtype=np.uint8)
    print(f"latih={len(raw):,} B tb={a.table_bits} match_bits={a.match_bits} "
          f"minlen={a.match_minlen} cks={a.cks_bits}\n", flush=True)

    for mfix, cfix in ((0, 0), (1, 1)):
        st = match_stats(raw, a.ntrain, a.table_bits, a.match_minlen, a.match_bits,
                         mfix, cfix, a.cks_bits)
        print(f"match_fix={mfix} ctx_fix={cfix}: akurasi-match={st['akurasi_match']:.3f} "
              f"cakupan={st['cakupan']:.3f} |w_match|={st['w_match']:.4f} "
              f"|w_tabel|={st['w_tabel']:.4f} |w_neural|={st['w_neural']:.4f}", flush=True)


if __name__ == "__main__":
    main()
