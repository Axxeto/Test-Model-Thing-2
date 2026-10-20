"""TINGKAT A — Inti statistik biner (rancangan TMT2 v2, ARCHITECTURE_V2.md §4.1).

Dekomposisi BINER: satu byte = 8 keputusan biner. Tiap konteks memberi p(bit), dan
masukan mixer adalah ``stretch(p) = ln(p/(1-p))``. Biaya per byte O(bit × input), bukan
O(V × orde) seperti mixer bertingkat-simbol lama.

Isi:
  * tabel hash terbuka (direct-mapped) dengan checksum 8-bit;
  * penghitung 16-bit dengan laju ADAPTIF (pemangkasan saat jenuh → cepat di konteks baru,
    lambat di konteks stabil);
  * konteks byte orde 1..6, 8, 12, 16, 24;
  * mixer satu lapis dengan set bobot dipilih oleh node-bit parsial (``c0``);
  * API ``step(byte, train=) -> CE bit``, ``eval_stream(data)``, prekuensial & bebas bocor.

Yang BELUM ada di sini (M2/M3 di dokumen): konteks jarang & konteks kata, model match
panjang/pendek, Mixer-2 berlapis, APM/SSE. Lihat ``ARCHITECTURE_V2.md`` §8.

Catatan kejujuran: modul ini adalah *acuan* yang harus lulus G1 (≥ +0,05 bit di atas mixer
bertingkat-simbol pada WikiText 200k/8k, 3 seed) SEBELUM dipakai. Jangan klaim apa pun
sebelum diukur.
"""
import math

import numpy as np
from numba import njit

# ---------------------------------------------------------------- konstanta
ORDERS = (1, 2, 3, 4, 5, 6, 8, 12, 16, 24)   # konteks byte (M1; M2 menambah jarang/kata)
HBUF = 64                                    # ukuran buffer riwayat (>= max(ORDERS))
NLEV = 4096                                  # level kuantisasi untuk tabel stretch
COUNT_CAP = 1023                             # ambang pemangkasan penghitung (uint16)
# Lebar checksum slot DEFAULT (bit). Layout tabel: (cksum << 32) | (n0 << 16) | n1.
#   8 bit (versi lama) -> hanya 256 nilai. Untuk tabel 2^25 slot, tiap slot "dipakai
#   bersama" ~2^25/2^8 = 131.072 konteks berbeda, sehingga sebuah slot hampir SELALU
#   cocok-checksum dengan konteks yang SALAH -> statistik tercampur acak.
#   24 bit -> 16,7 juta nilai; peluang tabrakan palsu turun ~65.000x.
# PENTING: ini hanya DEFAULT. Nilai yang benar-benar dipakai kernel diteruskan sebagai
# ARGUMEN `cks_bits` (cermin jebakan `COUNT_CAP` — konstanta global tidak bisa diubah
# runtime karena numba membekukannya saat compile).
CKS_BITS = 24
CK_MASK = (1 << CKS_BITS) - 1                # 8-bit lama: 255
FNV_OFF = np.uint64(14695981039346656037)
FNV_PRM = np.uint64(1099511628211)
GOLDEN = np.uint64(0x9E3779B97F4A7C15)
CCTX = np.uint64(0xD6E8FEB86659FD93)         # pengali indeks konteks (lihat catatan di stat_step)
M1_ = np.uint64(0xBF58476D1CE4E5B9)
M2_ = np.uint64(0x94D049BB133111EB)


@njit(inline="always", cache=True)
def _mix64(x):
    """Finalizer splitmix64 (x: uint64)."""
    x = (x ^ (x >> np.uint64(30))) * M1_
    x = (x ^ (x >> np.uint64(27))) * M2_
    return x ^ (x >> np.uint64(31))


@njit(inline="always", cache=True)
def _stretch(p, stab):
    i = int(p * NLEV)
    if i < 0:
        i = 0
    elif i > NLEV - 1:
        i = NLEV - 1
    return stab[i]


@njit(cache=True, fastmath=True)
def stat_step(byte, hist, pos, hlen, tab, mask, w1, w2, coff, clen, stab, lr, lr2, train,
              keys, st, apm1, apm2, alr, prev_byte, use_apm,
              mchk, mbyte, mmask, minlen, mconf, wsel_mask, probs, hctx, havail,
              nn_row, use_nn, count_cap, ck_mask, match_fix, ctx_fix, mck_mask,
              charclass, wsel_mode=0, stretch_exact=0, w_bound=0.0,
              diag=np.zeros(16), use_diag=0, rel=np.zeros(40),
              mixer_mode=0, s2_clip=16.0, mix_lr_floor=1.0, mix_lr_inv=0.0,
              mix_l2=0.0, cnt=np.zeros((3, 4096), dtype=np.int64),
              opt=np.zeros(128, dtype=np.int64),
              mrun=np.zeros(1, dtype=np.int64), smtab=np.full(32, 0.5),
              rngs=np.zeros(1, dtype=np.uint64)):
    """Parameter Mixer v3 (Fase 1; default = perilaku legacy bit-identik):

    ``mixer_mode`` 0=legacy (galat akhir global utk lapis-1, clamp ±16, lr tetap);
    1=galat-sendiri per set gaya PAQ8 + jadwal per-baris + tarikan L2.
    ``s2_clip`` batas input lapis-2 (mode 1; legacy = 16). ``mix_lr_floor`` lantai
    faktor jadwal; ``mix_lr_inv`` = 1/nhalf (0 = jadwal nonaktif); ``mix_l2``
    koefisien tarikan ke ``wref=1/nmix``; ``cnt`` hitungan kunjungan [3,4096].

    Parameter diagnostik (Fase 0, tak mengubah perilaku bila use_diag=0):

    ``    diag`` (16 float): [nbit, sat0, sat1, sat2, ownCE0, ownCE1, ownCE2,
    pmCE, pfCE, 0...]. ``rel`` (40 float): [cnt_bin0..19, pos_bin0..19] untuk
    kurva reliabilitas pm (20 bin). Hanya diakumulasi bila ``use_diag == 1``.
    Default array-nol menjaga kompatibilitas pemanggil lama (perilaku NOL berubah
    bila flag 0 — dijaga `test_diag_does_not_change_results`).

    ``opt`` (int64[64], default nol): flag fitur bertahap tanpa menambah argumen.
    opt[0]=1 -> bucket asosiatif 4-way + eviksi hitungan-terendah (#4).
    """
    """Satu byte: prediksi 8 bit (CE prekuensial), lalu perbarui tabel + mixer + APM + match.

    ``count_cap`` = ambang pemangkasan penghitung tabel (dulu konstanta modul ``COUNT_CAP``).
    ⚠️ DIJADIKAN PARAMETER 2026-10-04. Sebelumnya ia konstanta global yang dibaca numba saat
    COMPILE; menetapkan ``sc.COUNT_CAP = x`` di runtime TIDAK berpengaruh (terbukti: cap
    1023 vs 4095 memberi BPB identik 2,483178). Kalau diuji sebagai global, hasil "knob tidak
    menolong" akan PALSU. Sebagai argumen, nilainya benar-benar dipakai.

    ``hist`` = ring buffer byte terakhir; ``pos`` = jumlah byte yang sudah masuk.
    ``keys``/``st``/``hctx``/``havail`` = scratch (dipakai ulang, nol alokasi).
    ``apm1`` (256×33) dan ``apm2`` (65536×33) = peta SSE; ``use_apm=0`` mematikannya.
    ``mchk``/``mbyte``/``mmask``/``minlen`` = model match (M2): konteks minlen byte ->
    byte penerus pada kemunculan terakhir; ``mconf`` = besar sinyal stretch-nya.
    Mengembalikan CE dalam BIT.
    """
    nctx = clen.shape[0]
    nmix = nctx + 3                                 # + match + neural + bias
    c0 = 1
    ce = 0.0
    # ---- model match: cari kemunculan terakhir konteks minlen byte
    mpred = -1
    msl = -1
    mck = 0
    if hlen >= minlen:
        hm = FNV_OFF
        for j in range(minlen):
            hm = (hm ^ np.uint64(hist[(pos - 1 - j) & (HBUF - 1)])) * FNV_PRM
        mkey = _mix64(hm)
        msl = np.int64(mkey & np.uint64(mmask))
        # ⚠️ CACAT DIPERBAIKI 2026-10-04. Versi lama: mck = mkey & 255. Karena msl juga
        # memakai bit BAWAH (msl = mkey & mmask, mmask <= 2^24-1), maka mck SELALU sama
        # dengan (msl & 255) -> pengecekan mchk[msl]==mck selalu BENAR untuk slot terisi
        # -> model match tak pernah memverifikasi konteks (terbukti 100% dari 299.994
        # sampel). Akibat: slot match "cocok" dgn konteks acak apa pun -> prediksi palsu.
        # Perbaikan: checksum diambil dari bit yang TIDAK dipakai indeks (>= bit 40).
        # MATCH_FIX dijadikan argumen agar bisa dibandingkan terkontrol (0 = perilaku lama).
        if match_fix == 1:
            mck = np.int64((mkey >> np.uint64(40)) & np.uint64(mck_mask))
        else:
            mck = np.int64(mkey & np.uint64(255))
        if mchk[msl] == mck:
            mpred = np.int64(mbyte[msl])
    st[nmix - 1] = 1.0                              # bias
    # ---- HASH KONTEKS: dihitung SEKALI per byte. Konteks byte TIDAK bergantung pada bit,
    #      jadi sebelumnya kita menghitung hash yang sama 8x (sekali per bit). Mengangkatnya
    #      keluar loop bit menghapus ~7/8 pekerjaan hashing — bagian terbesar biaya per byte.
    for c in range(nctx):
        if coff[c, 0] == -2:
            wl = 0
            while wl < MAXWORD and wl < hlen:
                if not _isalpha(hist[(pos - 1 - wl) & (HBUF - 1)]):
                    break
                wl += 1
            if wl == 0:
                havail[c] = 0
                continue
            h = FNV_OFF
            for j in range(wl):
                h = (h ^ np.uint64(hist[(pos - 1 - j) & (HBUF - 1)])) * FNV_PRM
            hctx[c] = h
            havail[c] = 1
        else:
            k = clen[c]
            if k > 0 and hlen <= coff[c, k - 1]:
                havail[c] = 0
                continue
            h = FNV_OFF                          # k == 0 -> konteks kosong (order-0)
            for j in range(k):
                h = (h ^ np.uint64(hist[(pos - 1 - coff[c, j]) & (HBUF - 1)])) * FNV_PRM
            hctx[c] = h
            havail[c] = 1
    # ---- DROPOUT konteks saat latih (regularizer robustness ala adapter-m-disampel).
    # Hanya train==1 dan opt[9]>0 (permille). Orde 1-4 + kata + match/NN/bias TAK
    # boleh drop. Sekali per byte (konstan di 8 bit). LCG di rngs[0] (state model).
    if train == 1 and opt[9] > 0:
        for c in range(nctx):
            if havail[c] == 0:
                continue
            if clen[c] < 5:
                continue
            if coff[c, 0] == -2:                 # konteks kata: lindungi
                continue
            rngs[0] = rngs[0] * np.uint64(6364136223846793005) + np.uint64(1442695040888963407)
            if np.int64((rngs[0] >> np.uint64(11)) % np.uint64(1000)) < opt[9]:
                havail[c] = 0
    # ---- magnitudo match: konstanta (legacy) atau adaptif per mrun (C2-L1).
    # sm[k] = P(match benar | run k), init 0.5 (netral -> stretch ~0).
    if opt[10] == 1:
        kk = mrun[0] if mrun[0] < 31 else 31
        mag = _stretch(smtab[kk], stab)
    else:
        mag = mconf
    for bi in range(8):
        bit = (byte >> (7 - bi)) & 1
        # ---- kumpulkan prediksi tiap konteks (hash sudah siap dari luar loop bit)
        maxhit = 0
        for c in range(nctx):
            if havail[c] == 0:
                st[c] = 0.0
                continue
            # ⚠️ CACAT DIPERBAIKI 2026-10-04. Versi lama TIDAK menyertakan indeks konteks `c`
            # di hash: `key = _mix64(hctx[c] ^ (c0*GOLDEN))`. Akibatnya dua konteks BERBEDA
            # yang kebetulan punya byte sama (mis. konteks kata vs konteks orde-k) memetakan
            # ke slot YANG SAMA dan saling menimpa (ditulis 2x per bit, yang terakhir menang)
            # -> kapasitas efektif berkurang. Perbaikan: campurkan `c` ke hash.
            # CTX_FIX dijadikan argumen agar bisa dibandingkan terkontrol (0 = perilaku lama).
            if ctx_fix == 1:
                key = _mix64(hctx[c] ^ (np.uint64(c0) * GOLDEN)
                             ^ (np.uint64(c + 1) * CCTX))
            else:
                key = _mix64(hctx[c] ^ (np.uint64(c0) * GOLDEN))
            keys[c] = key                            # simpan KUNCI (bukan slot+cks): separuh trafik
            sl = np.int64(key & np.uint64(mask))
            ck = np.int64((key >> np.uint64(32)) & np.uint64(ck_mask))
            d, sl = _tab_lookup(tab, mask, key, ck, opt[0])
            if (d >> 32) == ck:
                a = (d >> 16) & 65535
                b = d & 65535
                st[c] = _stretch((b + 0.5) / (a + b + 1.0), stab)
                if clen[c] > maxhit:
                    maxhit = clen[c]
            else:
                st[c] = 0.0
        # ---- masukan match
        if mpred >= 0:
            mb = (mpred >> (7 - bi)) & 1
            st[nctx] = mag if mb == 1 else -mag
        else:
            st[nctx] = 0.0
        # ---- masukan NEURAL (Tingkat B): stretch(p(bit|prefiks)) yang sudah dihitung di luar
        st[nctx + 1] = nn_row[bi] if use_nn == 1 else 0.0
        # ---- MIXER 2 LAPIS (M3). Lapis 1: 3 mixer dengan pemilih BERBEDA.
        #      Catatan penting: stretch(squash(d)) == d, jadi keluaran lapis-1 TIDAK perlu
        #      exp sama sekali — cukup logitnya. Semua mixer lapis-1 dilatih dengan galat
        #      AKHIR (gaya lpaq), sehingga hanya SATU exp per bit yang dibayar.
        if opt[1] == 1:
            # Selector set-0 per hit-order (A2): orde-tertinggi yang hit
            # (miss/1-3->0; 4-6->1; 8-12->2; 16-24->3). Butuh nctx<=32 (assert).
            hb = 3 if maxhit >= 16 else (2 if maxhit >= 8 else (1 if maxhit >= 4 else 0))
            sel0 = (((hb & 3) << 8) | (c0 & 255)) & wsel_mask
        else:
            sel0 = c0 & 255
        sel1 = prev_byte & 255
        if wsel_mode == 1:
            # B-02: kelas karakter prev_byte (4 bit cukup: 16 kelas) + c0 penuh
            sel2 = ((int(charclass[prev_byte & 255]) << 8) | c0) & wsel_mask
        else:
            sel2 = ((prev_byte << 8) | c0) & wsel_mask
        d0 = 0.0
        d1 = 0.0
        d2 = 0.0
        for c in range(nmix):
            x = st[c]
            d0 += w1[0, sel0, c] * x
            d1 += w1[1, sel1, c] * x
            d2 += w1[2, sel2, c] * x
        if mixer_mode == 1:
            # Mixer v3: prediksi SENDIRI per set (regresi logistik konveks per baris).
            # Clip ±20 agar squash tak underflow; input lapis-2 dibatasi s2_clip.
            p0 = 1.0 / (1.0 + math.exp(-_clip20(d0)))
            p1 = 1.0 / (1.0 + math.exp(-_clip20(d1)))
            p2 = 1.0 / (1.0 + math.exp(-_clip20(d2)))
            s20 = d0
            if s20 < -s2_clip:
                s20 = -s2_clip
            elif s20 > s2_clip:
                s20 = s2_clip
            s21 = d1
            if s21 < -s2_clip:
                s21 = -s2_clip
            elif s21 > s2_clip:
                s21 = s2_clip
            s22 = d2
            if s22 < -s2_clip:
                s22 = -s2_clip
            elif s22 > s2_clip:
                s22 = s2_clip
        else:
            p0 = 0.0
            p1 = 0.0
            p2 = 0.0
            s20 = d0
            s21 = d1
            s22 = d2
            if s20 < -16.0:
                s20 = -16.0
            elif s20 > 16.0:
                s20 = 16.0
            if s21 < -16.0:
                s21 = -16.0
            elif s21 > 16.0:
                s21 = 16.0
            if s22 < -16.0:
                s22 = -16.0
            elif s22 > 16.0:
                s22 = 16.0
        selL = c0 & 255
        dot = (w2[selL, 0] * s20 + w2[selL, 1] * s21 + w2[selL, 2] * s22 + w2[selL, 3])
        pm = 1.0 / (1.0 + math.exp(-dot))
        # ---- APM / SSE dua tahap (refinement)
        pf = pm
        i1 = 0
        wf = 0.0
        c2 = 0
        if use_apm == 1:
            sm = _stretch_clamped(pm, stab, stretch_exact)
            x = (sm + 8.0) * 2.0
            if x < 0.0:
                x = 0.0
            elif x > 31.0:
                x = 31.0
            i1 = int(x)
            wf = x - i1
            pa1 = apm1[c0, i1] * (1.0 - wf) + apm1[c0, i1 + 1] * wf
            c2 = (prev_byte << 8) | c0
            pa2 = apm2[c2, i1] * (1.0 - wf) + apm2[c2, i1 + 1] * wf
            pf = (pm + pa1 + pa2) / 3.0
        if pf < 1e-6:
            pf = 1e-6
        elif pf > 1.0 - 1e-6:
            pf = 1.0 - 1e-6
        probs[bi] = pf
        if use_diag == 1:
            # --- diagnostik Fase 0 (M1/M2): saturasi, CE-sendiri per set, reliabilitas
            diag[0] += 1.0
            if d0 > 16.0 or d0 < -16.0:
                diag[1] += 1.0
            if d1 > 16.0 or d1 < -16.0:
                diag[2] += 1.0
            if d2 > 16.0 or d2 < -16.0:
                diag[3] += 1.0
            p0 = 1.0 / (1.0 + math.exp(-s20))
            p1 = 1.0 / (1.0 + math.exp(-s21))
            p2 = 1.0 / (1.0 + math.exp(-s22))
            if bit == 1:
                diag[4] += -math.log2(p0)
                diag[5] += -math.log2(p1)
                diag[6] += -math.log2(p2)
                diag[7] += -math.log2(pm)
            else:
                diag[4] += -math.log2(1.0 - p0)
                diag[5] += -math.log2(1.0 - p1)
                diag[6] += -math.log2(1.0 - p2)
                diag[7] += -math.log2(1.0 - pm)
            diag[8] += (-math.log2(pf) if bit == 1 else -math.log2(1.0 - pf))
            bb = int(pf * 20.0)
            if bb > 19:
                bb = 19
            rel[bb] += 1.0
            if bit == 1:
                rel[bb + 20] += 1.0
        if bit == 1:
            ce += -math.log2(pf)
        else:
            ce += -math.log2(1.0 - pf)
        # ---- perbarui mixer + APM + penghitung
        if train:
            err = bit - pm
            w2[selL, 0] += lr2 * err * s20
            w2[selL, 1] += lr2 * err * s21
            w2[selL, 2] += lr2 * err * s22
            w2[selL, 3] += lr2 * err
            if mixer_mode == 1:
                # Galat MILIK set sendiri + jadwal per-baris + tarikan L2.
                # Tiap baris = regresi logistik konveks -> minimizer terbatas,
                # set jenuh-yang-salah mendapat gaya pemulih (tak ada clamp-ghost).
                e0 = bit - p0
                e1 = bit - p1
                e2 = bit - p2
                f0 = 1.0 / (1.0 + cnt[0, sel0] * mix_lr_inv)
                if f0 < mix_lr_floor:
                    f0 = mix_lr_floor
                f1 = 1.0 / (1.0 + cnt[1, sel1] * mix_lr_inv)
                if f1 < mix_lr_floor:
                    f1 = mix_lr_floor
                f2 = 1.0 / (1.0 + cnt[2, sel2] * mix_lr_inv)
                if f2 < mix_lr_floor:
                    f2 = mix_lr_floor
                g0 = lr * f0
                g1 = lr * f1
                g2 = lr * f2
                if cnt[0, sel0] < 1152921504606846976:
                    cnt[0, sel0] += 1
                if cnt[1, sel1] < 1152921504606846976:
                    cnt[1, sel1] += 1
                if cnt[2, sel2] < 1152921504606846976:
                    cnt[2, sel2] += 1
                wref = 1.0 / float(nmix)
                for c in range(nmix):
                    x = st[c]
                    w1[0, sel0, c] += g0 * (e0 * x - mix_l2 * (w1[0, sel0, c] - wref))
                    w1[1, sel1, c] += g1 * (e1 * x - mix_l2 * (w1[1, sel1, c] - wref))
                    w1[2, sel2, c] += g2 * (e2 * x - mix_l2 * (w1[2, sel2, c] - wref))
            else:
                for c in range(nmix):
                    x = st[c]
                    w1[0, sel0, c] += lr * err * x
                    w1[1, sel1, c] += lr * err * x
                    w1[2, sel2, c] += lr * err * x
            # ★ PEMBATAS BOBOT (fix kurva-U, 2026-10-06). E1+E2 membuktikan penyebab kurva-U
            # adalah bobot mixer lapis-1 yang DIVERGEN tanpa batas (max|w1| 1,39 -> 1676 selama
            # 30 jt byte, lalu mendominasi keluaran -> BPB naik). `w_bound > 0` meng-clip tiap
            # bobot ke [-w_bound, +w_bound] setelah pembaruan; `w_bound == 0` = perilaku lama.
            if w_bound > 0.0:
                for c in range(nmix):
                    v = w1[0, sel0, c]
                    if v > w_bound:
                        w1[0, sel0, c] = w_bound
                    elif v < -w_bound:
                        w1[0, sel0, c] = -w_bound
                    v = w1[1, sel1, c]
                    if v > w_bound:
                        w1[1, sel1, c] = w_bound
                    elif v < -w_bound:
                        w1[1, sel1, c] = -w_bound
                    v = w1[2, sel2, c]
                    if v > w_bound:
                        w1[2, sel2, c] = w_bound
                    elif v < -w_bound:
                        w1[2, sel2, c] = -w_bound
            if use_apm == 1:
                apm1[c0, i1] += alr * (bit - apm1[c0, i1]) * (1.0 - wf)
                apm1[c0, i1 + 1] += alr * (bit - apm1[c0, i1 + 1]) * wf
                apm2[c2, i1] += alr * (bit - apm2[c2, i1]) * (1.0 - wf)
                apm2[c2, i1 + 1] += alr * (bit - apm2[c2, i1 + 1]) * wf
            for c in range(nctx):
                if havail[c] == 0:
                    continue
                key = keys[c]
                sl = np.int64(key & np.uint64(mask))
                ck = np.int64((key >> np.uint64(32)) & np.uint64(ck_mask))
                # 4-way: pindai ulang grup (kebenaran > kecepatan; Patch B
                # pemindaian-tunggal GUGUR terukur −0,065 — dirty-scan lebih
                # mahal dari rescan. Lihat HASIL_E7B_V3.md).
                sl = _tab_victim(tab, mask, key, ck, opt[0])
                d = tab[sl]
                if (d >> np.int64(32)) != ck:
                    a = 0
                    b = 0
                else:
                    a = (d >> 16) & 65535
                    b = d & 65535
                if (d >> 32) != ck:
                    a = 0
                    b = 0
                else:
                    a = (d >> 16) & 65535
                    b = d & 65535
                if bit == 1:
                    b += 1
                else:
                    a += 1
                if a + b > count_cap:               # laju adaptif: pangkas saat jenuh
                    a = a >> 1
                    b = b >> 1
                tab[sl] = (ck << 32) | (a << 16) | b
        if mpred >= 0 and ((mpred >> (7 - bi)) & 1) != bit:
            mpred = -1                              # match pecah -> matikan sisa byte
        c0 = c0 * 2 + bit
    # ---- perbarui indeks match (konteks SEBELUM byte ini -> byte ini)
    if train and msl >= 0:
        mchk[msl] = mck
        mbyte[msl] = byte
    # ---- C2-L1: run-length + keyakinan adaptif (hanya saat latih; eval beku).
    # correct = mpred bertahan 8 bit. sm[k] += (correct - sm[k])/64.
    if train == 1 and opt[10] == 1 and msl >= 0:
        correct = 1.0 if mpred >= 0 else 0.0
        kk2 = mrun[0] if mrun[0] < 31 else 31
        smtab[kk2] += (correct - smtab[kk2]) / 64.0
        mrun[0] = mrun[0] + 1 if mpred >= 0 else 0
    return ce


@njit(cache=True, fastmath=True)
def stat_stream(data, hist, pos, hlen, tab, mask, w1, w2, coff, clen, stab, lr, lr2, train,
                keys, st, apm1, apm2, alr, prev_byte, use_apm,
                mchk, mbyte, mmask, minlen, mconf, wsel_mask, probs, hctx, havail,
                nn_row, use_nn, count_cap, ck_mask, match_fix, ctx_fix, mck_mask,
                charclass, wsel_mode, stretch_exact, out=None, w_bound=0.0,
                mixer_mode=0, s2_clip=16.0, mix_lr_floor=1.0, mix_lr_inv=0.0,
                mix_l2=0.0, cnt=np.zeros((3, 4096), dtype=np.int64),
                opt=np.zeros(128, dtype=np.int64),
                mrun=np.zeros(1, dtype=np.int64), smtab=np.full(32, 0.5),
                rngs=np.zeros(1, dtype=np.uint64), nn_stream=None):
    """Proses SELURUH aliran byte dalam SATU panggilan numba (menghapus pajak loop Python).

    Ini pembungkus tipis di atas ``stat_step``: loop byte ada di dalam kode terkompilasi,
    bukan di Python. Mengembalikan array CE (bit) sepanjang ``len(data)``; entri pertama
    (byte saat ``hlen == 0``) bernilai ``nan`` — pemanggil harus mem-``nanmean``.

    ``nn_stream`` (opsional, (len(data), 8)): masukan guru eksternal per byte (stretch
    log-odds per bit) — dipakai bila ``use_nn == 1``. Menyalin baris ke ``nn_row`` di dalam
    loop supaya jalur cepat bisa memakai guru (mis. dari `tmt2/byte_bridge.py`) tanpa
    kembali ke loop Python. ``None`` = tak ada guru (perilaku lama, bit-identik).

    ⚡ Alasan keberadaan: memanggil ``StatCore.step`` per byte dari Python membatasi laju
    ke ~90 kB/s hanya karena overhead pemanggilan Python per byte; loop terkompilasi di sini
    mencapai laju kernel penuh. Sekaligus jadi jalur ``train_stream``/``eval_stream`` cepat.
    """
    n = data.shape[0]
    ce = np.empty(n, dtype=np.float64)
    tr = 1 if train else 0
    _dd = np.zeros(16)
    _rr = np.zeros(40)
    for i in range(n):
        if hlen == 0:
            # butuh riwayat minimal: hanya dorong byte ke ring buffer, CE tak terdefinisi
            hist[pos & (HBUF - 1)] = data[i]
            pos += 1
            if hlen < HBUF:
                hlen += 1
            prev_byte = data[i]
            ce[i] = np.nan
            continue
        if nn_stream is not None:            # guru eksternal: salin baris byte ini
            if use_nn == 1:
                for k in range(8):
                    nn_row[k] = nn_stream[i, k]
        ce[i] = stat_step(data[i], hist, pos, hlen, tab, mask, w1, w2, coff, clen, stab,
                          lr, lr2, tr, keys, st, apm1, apm2, alr, prev_byte,
                          use_apm, mchk, mbyte, mmask, minlen, mconf, wsel_mask, probs,
                          hctx, havail, nn_row, use_nn, count_cap, ck_mask, match_fix,
                          ctx_fix, mck_mask, charclass, wsel_mode, stretch_exact, w_bound,
                          _dd, 0, _rr,
                          mixer_mode, s2_clip, mix_lr_floor, mix_lr_inv, mix_l2, cnt,
                          opt, mrun, smtab, rngs)
        hist[pos & (HBUF - 1)] = data[i]
        pos += 1
        if hlen < HBUF:
            hlen += 1
        prev_byte = data[i]
    if out is not None:
        out[0] = pos
        out[1] = hlen
        out[2] = prev_byte
    return ce


@njit(cache=True, fastmath=True)
def stat_diagnose(data, hist, pos, hlen, tab, mask, w1, w2, coff, clen, stab, lr, lr2,
                  keys, st, apm1, apm2, alr, prev_byte, use_apm,
                  mchk, mbyte, mmask, minlen, mconf, wsel_mask, probs, hctx, havail,
                  nn_row, use_nn, count_cap, ck_mask, match_fix, ctx_fix, mck_mask,
                  charclass, wsel_mode, stretch_exact, w_bound, diag, rel,
                  mixer_mode=0, s2_clip=16.0, mix_lr_floor=1.0, mix_lr_inv=0.0,
                  mix_l2=0.0, vst=np.zeros((3, 4096), dtype=np.int64),
                  opt=np.zeros(128, dtype=np.int64),
                  mrun=np.zeros(1, dtype=np.int64), smtab=np.full(32, 0.5),
                  rngs=np.zeros(1, dtype=np.uint64)):
    """Jalankan aliran dengan train=0 (TIDAK mengubah tabel/mixer/APM/match) sambil
    mengakumulasi diagnostik Fase 0 ke ``diag``/``rel``. hist/pos ikut maju (konteks
    berjalan) — pemanggil yang butuh konteks utuh harus snapshot & restore."""
    ce = 0.0
    for i in range(data.shape[0]):
        if hlen == 0:
            hist[pos & (HBUF - 1)] = data[i]
            pos += 1
            if hlen < HBUF:
                hlen += 1
            prev_byte = data[i]
            continue
        ce += stat_step(data[i], hist, pos, hlen, tab, mask, w1, w2, coff, clen, stab,
                        lr, lr2, 0, keys, st, apm1, apm2, alr, prev_byte,
                        use_apm, mchk, mbyte, mmask, minlen, mconf, wsel_mask, probs,
                        hctx, havail, nn_row, use_nn, count_cap, ck_mask, match_fix,
                        ctx_fix, mck_mask, charclass, wsel_mode, stretch_exact, w_bound,
                        diag, 1, rel,
                        mixer_mode, s2_clip, mix_lr_floor, mix_lr_inv, mix_l2, vst,
                        opt, mrun, smtab, rngs)
        hist[pos & (HBUF - 1)] = data[i]
        pos += 1
        if hlen < HBUF:
            hlen += 1
        prev_byte = data[i]
    return ce


@njit(cache=True, fastmath=True)
def stat_probe_hitrate(keys, havail, tab, mask, ck_mask, st, nctx, hr, av):
    """Isi ``hr`` (hit checksum per konteks) & ``av`` (|stretch| per konteks) dari state
    SESAAT INI — dipakai E2 pada loop terkompilasi (bukan loop Python per byte)."""
    for cc in range(nctx):
        hr[cc] = 0.0
        av[cc] = 0.0
    for cc in range(nctx):
        if havail[cc] == 0:
            continue
        key = keys[cc]
        sl = np.int64(key & np.uint64(mask))
        ck = np.int64((key >> np.uint64(32)) & np.uint64(ck_mask))
        if (tab[sl] >> 32) == ck:
            hr[cc] = 1.0
    for cc in range(nctx):
        av[cc] = abs(st[cc])


@njit(cache=True, fastmath=True)
def stat_instrument(data, interval, probe_bytes, hist, pos, hlen, tab, mask, w1, w2, coff,
                    clen, stab, lr, lr2, keys, st, apm1, apm2, alr, prev_byte, use_apm,
                    mchk, mbyte, mmask, minlen, mconf, wsel_mask, probs, hctx, havail,
                    nn_row, use_nn, count_cap, ck_mask, match_fix, ctx_fix, mck_mask,
                    charclass, wsel_mode, stretch_exact, out_hr, out_av, out_idx,
                    out_w1mx, out_w1mn, out_w2mn, out_fill, out=None, w_bound=0.0,
                    mixer_mode=0, s2_clip=16.0, mix_lr_floor=1.0, mix_lr_inv=0.0,
                    mix_l2=0.0, vst=np.zeros((3, 4096), dtype=np.int64),
                    opt=np.zeros(128, dtype=np.int64),
                    mrun=np.zeros(1, dtype=np.int64), smtab=np.full(32, 0.5),
                    rngs=np.zeros(1, dtype=np.uint64)):
    """E2 TERKOMPILASI: latih aliran, tiap ``interval`` byte jalankan probe ``probe_bytes``
    tanpa belajar (state hist/pos dipulihkan), lalu tulis hit-rate/|stretch| per konteks ke
    ``out_hr``/``out_av`` pada baris ``out_idx``. Mengembalikan jumlah probe yang tertulis.

    CATATAN PROTOKOL: ``pos``/``hlen``/``prev_byte`` dipulihkan setelah probe, tetapi
    ``tab``/``w1``/``w2``/``apm`` TIDAK berubah selama probe (train=0) → hasil latih identik
    dengan tanpa instrumentasi. Ini menjaga paritas dengan jalur produksi.
    """
    n = data.shape[0]
    nctx = clen.shape[0]
    nmix = nctx + 3
    nprobe = out_hr.shape[0]
    hbuf = np.zeros(HBUF, dtype=np.uint8)      # cadangan hist selama probe
    smbuf = np.zeros(32, dtype=np.float64)     # cadangan peta sm selama probe
    _dd = np.zeros(16)
    _rr = np.zeros(40)
    k = 0
    i = 0
    while i < n:
        end = i + interval
        if end > n:
            end = n
        for j in range(i, end):
            b = data[j]
            if hlen > 0:
                stat_step(b, hist, pos, hlen, tab, mask, w1, w2, coff, clen, stab, lr, lr2,
                          1, keys, st, apm1, apm2, alr, prev_byte, use_apm, mchk, mbyte,
                          mmask, minlen, mconf, wsel_mask, probs, hctx, havail, nn_row,
                          0, count_cap, ck_mask, match_fix, ctx_fix, mck_mask, charclass,
                          wsel_mode, stretch_exact, w_bound, _dd, 0, _rr,
                          mixer_mode, s2_clip, mix_lr_floor, mix_lr_inv, mix_l2, vst,
                          opt, mrun, smtab, rngs)
            hist[pos & (HBUF - 1)] = b
            pos += 1
            if hlen < HBUF:
                hlen += 1
            prev_byte = b
        pe = end + probe_bytes
        if pe > n:
            pe = n
        if pe <= end:
            break
        sv_pos = pos
        sv_prev = prev_byte
        sv_hlen = hlen
        sv_mrun = mrun[0]                      # + state adaptif (dropout/match)
        sv_rng = rngs[0]
        for c in range(HBUF):                  # cadangkan ring buffer (agar probe tak bocor)
            hbuf[c] = hist[c]
        for c in range(32):                    # cadangkan peta keyakinan match
            smbuf[c] = smtab[c]
        hr = np.zeros(nctx, dtype=np.float64)
        av = np.zeros(nctx, dtype=np.float64)
        cnt = 0
        for j in range(end, pe):
            stat_probe_hitrate(keys, havail, tab, mask, ck_mask, st, nctx, hr, av)
            for c in range(nctx):
                out_hr[k, c] += hr[c]
                out_av[k, c] += av[c]
            cnt += 1
            b = data[j]
            if hlen > 0:
                stat_step(b, hist, pos, hlen, tab, mask, w1, w2, coff, clen, stab, lr, lr2,
                          0, keys, st, apm1, apm2, alr, prev_byte, use_apm, mchk, mbyte,
                          mmask, minlen, mconf, wsel_mask, probs, hctx, havail, nn_row,
                          0, count_cap, ck_mask, match_fix, ctx_fix, mck_mask, charclass,
                          wsel_mode, stretch_exact, w_bound, _dd, 0, _rr,
                          mixer_mode, s2_clip, mix_lr_floor, mix_lr_inv, mix_l2, vst,
                          opt, mrun, smtab, rngs)
            hist[pos & (HBUF - 1)] = b
            pos += 1
            if hlen < HBUF:
                hlen += 1
            prev_byte = b
        # pulihkan keadaan stream TEPAT seperti akhir interval latih: pos/prev/hlen DAN
        # isi ring buffer (byte probe sempat menimpa hist -> harus dikembalikan).
        pos = sv_pos
        prev_byte = sv_prev
        hlen = sv_hlen
        mrun[0] = sv_mrun
        rngs[0] = sv_rng
        for c in range(HBUF):
            hist[c] = hbuf[c]
        for c in range(32):
            smtab[c] = smbuf[c]
        if cnt > 0:
            for c in range(nctx):
                out_hr[k, c] /= cnt
                out_av[k, c] /= cnt
        # norma mixer & isi tabel PADA titik ini (bukan di akhir seluruh latih)
        wmx = 0.0
        wsum = 0.0
        wcnt = 0
        for a3 in range(w1.shape[0]):
            for s in range(w1.shape[1]):
                for c in range(w1.shape[2]):
                    v = abs(w1[a3, s, c])
                    if v > wmx:
                        wmx = v
                    wsum += v
                    wcnt += 1
        w2sum = 0.0
        for r in range(w2.shape[0]):
            for c in range(w2.shape[1]):
                w2sum += abs(w2[r, c])
        nz = 0
        for s in range(tab.shape[0]):
            if tab[s] != 0:
                nz += 1
        out_w1mx[k] = wmx
        out_w1mn[k] = wsum / max(wcnt, 1)
        out_w2mn[k] = w2sum / max(w2.shape[0] * w2.shape[1], 1)
        out_fill[k] = nz / tab.shape[0]
        out_idx[k] = end
        k += 1
        i = end
    if out is not None:
        out[0] = pos
        out[1] = hlen
        out[2] = prev_byte
    return k


@njit(inline="always", cache=True)
def _stretch_exact(p):
    """Stretch FLOAT murni (tanpa kuantisasi tabel) — perbaikan B-07.

    Tabel `stab` mengkuantisasi p ke 1/4096; di ekor (p≈0,999) galat odds bisa ~25%, tepat
    di wilayah teks repetitif tempat bit termurah. Untuk APM, hitung logit eksak.
    """
    if p < 1e-7:
        p = 1e-7
    elif p > 1.0 - 1e-7:
        p = 1.0 - 1e-7
    return math.log(p / (1.0 - p))


@njit(inline="always", cache=True)
def _stretch_clamped(p, stab, use_exact=0):
    if use_exact == 1:
        return _stretch_exact(p)
    return _stretch(p, stab)


@njit(inline="always", cache=True)
def _squash(x):
    return 1.0 / (1.0 + math.exp(-x))


@njit(inline="always", cache=True)
def _tab_lookup(tab, mask, key, ck, assoc):
    """Lookup tabel: direct-mapped (assoc=0) atau grup 4-way (assoc=1).

    Mengembalikan (d, sl): isi slot + indeks slot EKSAK (untuk 4-way = slot
    yang cocok, atau grup & ~3 bila tak cocok — update memindai ulang).
    Slot kosong (0) cocok bila ck==0 -> a=b=0 (sama seperti legacy).
    """
    sl = np.int64(key & np.uint64(mask))
    if assoc == 0:
        return tab[sl], sl
    g = sl & np.int64(-4)
    for j in range(4):
        dj = tab[g + j]
        if (dj >> np.int64(32)) == ck:
            return dj, g + j
    return np.int64(0), g


@njit(inline="always", cache=True)
def _tab_victim(tab, mask, key, ck, assoc):
    """Slot tulis: cocok-cek (ada -> slot itu) atau victim min(n0+n1)."""
    sl = np.int64(key & np.uint64(mask))
    if assoc == 0:
        return sl
    g = sl & np.int64(-4)
    for j in range(4):
        if (tab[g + j] >> np.int64(32)) == ck:
            return g + j
    best = np.int64(1 << 62)
    vsl = g
    for j in range(4):
        dj = tab[g + j]
        tot = ((dj >> np.int64(16)) & np.int64(65535)) + (dj & np.int64(65535))
        if tot < best:
            best = tot
            vsl = g + j
    return vsl


@njit(inline="always", cache=True)
def _clip20(x):
    if x < -20.0:
        return -20.0
    elif x > 20.0:
        return 20.0
    return x


@njit(inline="always", cache=True)
def _isalpha(c):
    return (65 <= c <= 90) or (97 <= c <= 122)


MAXK = 26          # panjang konteks maksimum (offset 0..25)
MAXWORD = 16       # panjang maksimum konteks kata
SPARSE_SPECS = ((1, 2), (0, 2), (1, 3), (0, 3), (1, 2, 3), (0, 1, 3))


def build_contexts(orders, sparse=True, word=True):
    """Bangun matriks offset konteks.

    Setiap konteks = daftar offset ke belakang (0 = byte terakhir). Orde-``k`` dense = (0..k-1).
    Konteks JARANG melewati byte tertentu (mis. (1,2) = byte t-2 dan t-3 tanpa t-1) — menangkap
    pola yang tak terjangkau orde dense. Konteks KATA = huruf sejak pembatas terakhir, ditandai
    ``coff[c, 0] = -1``.
    """
    specs = [tuple(range(int(k))) for k in orders]
    if sparse:
        specs += list(SPARSE_SPECS)
    n = len(specs) + (1 if word else 0)
    coff = np.full((n, MAXK), -1, dtype=np.int64)
    clen = np.zeros(n, dtype=np.int64)
    for i, s in enumerate(specs):
        clen[i] = len(s)
        for j, o in enumerate(s):
            coff[i, j] = o
    if word:
        coff[n - 1, 0] = -2                   # sentinel: -2 = konteks kata; -1 = tak dipakai
        clen[n - 1] = 0
    return coff, clen


def build_stretch(n=NLEV):
    p = (np.arange(n) + 0.5) / n
    return np.log(p / (1.0 - p)).astype(np.float64)


def build_charclass():
    """Kelas karakter 0..15 untuk pemilih mixer (perbaikan B-02).

    Pemilih lama `((prev_byte<<8)|c0)&wsel_mask` pada wsel_bits=12 hanya menyisakan 4 bit
    RENDAH prev_byte -> 'a','q','A' semua jatuh ke pemilih yang sama (terbukti). Kelas
    karakter memberi partisi yang SEMANTIK: huruf kecil (pecah per-vokal/konsonan umum),
    huruf besar, digit, spasi, tanda baca umum, newline, biner/UTF-8, dll.
    """
    cls = np.full(256, 15, dtype=np.int64)   # default: lain-lain/bit-tinggi
    for b in range(32, 127):                 # ASCII tercetak
        ch = chr(b)
        if "aeiou".find(ch) >= 0:
            cls[b] = 0                       # vokal
        elif ch.isalpha():
            cls[b] = 1                       # konsonan
        elif ch.isdigit():
            cls[b] = 2
        elif ch in ".,;:!?":
            cls[b] = 3
        elif ch in "\"'()[]{}":
            cls[b] = 4
        elif ch in "-_/\\@#&*+=<>|~^$%":
            cls[b] = 5
        else:
            cls[b] = 6
    cls[9] = 7                               # tab
    cls[10] = 8                              # newline
    cls[13] = 9                              # CR
    cls[32] = 10                             # spasi
    for b in range(128, 256):                # byte tinggi: UTF-8/aksen
        cls[b] = 11 if b >= 192 else 12
    cls[0] = 13
    cls[255] = 14
    return cls


APM_BUCKETS = 33


def build_apm(nctx):
    """Peta APM awal = identitas: bucket i memetakan ke squash(i/2 - 8) (kebalikan stretch)."""
    x = np.arange(APM_BUCKETS) / 2.0 - 8.0
    p = 1.0 / (1.0 + np.exp(-x))
    return np.tile(p, (nctx, 1)).astype(np.float64)


class StatCore:
    """Inti statistik biner (Tingkat A). Prekuensial, online, tabel berarray datar."""

    def __init__(self, orders=ORDERS, table_bits=22, lr=0.002, seed=0,
                 use_apm=True, apm_lr=0.02,
                 match_minlen=6, match_bits=20, match_conf=2.0, wsel_bits=8, lr2=3e-3,
                 sparse=True, word=True, count_cap=COUNT_CAP, cks_bits=CKS_BITS,
                 match_fix=True, ctx_fix=True, match_ck_bits=8, wsel_mode=0, stretch_exact=0,
                 w_bound=0.0, mixer_freeze_after=0,
                 mixer_mode=0, s2_clip=16.0, mix_lr_floor=1.0, mix_lr_nhalf=0,
                 mix_l2=0.0, assoc=0, fast_scan=0, hit_sel=0,
                 dropout_permille=0, match_adapt=0):
        self.orders = np.asarray(orders, dtype=np.int64)
        self._sparse = bool(sparse)
        self._word = bool(word)
        self.coff, self.clen = build_contexts(self.orders, sparse=sparse, word=word)
        self.nctx = int(self.clen.shape[0])
        size = 1 << int(table_bits)
        self.mask = size - 1
        # SATU array int64 per slot: (checksum << 32) | (n0 << 16) | n1.
        # Sebelumnya 3 array terpisah -> 3 akses memori acak per lookup. Karena tabelnya
        # besar (ratusan MB), akses acak inilah hambatan utama kecepatan.
        self.tab = np.zeros(size, dtype=np.int64)
        self.wsel_bits = int(wsel_bits)
        self.wsel_mask = (1 << self.wsel_bits) - 1
        # B-02: mode pemilih set-2. 0 = lama ((prev_byte<<8)|c0 — hanya 4 bit rendah prev),
        # 1 = kelas-karakter (prev_class<<8)|c0 — partisi semantik.
        self.wsel_mode = int(wsel_mode)
        self.charclass = build_charclass()
        # B-07: stretch eksak (float) untuk APM, bukan tabel terkuantisasi 1/4096
        self.stretch_exact = 1 if stretch_exact else 0
        # fix kurva-U: batas |bobot| mixer lapis-1 (0 = tanpa batas = perilaku lama)
        self.w_bound = float(w_bound)
        # Mixer v3 (Fase 1): 0 = legacy bit-identik; 1 = galat-sendiri + jadwal + L2.
        self.mixer_mode = int(mixer_mode)
        self.s2_clip = float(s2_clip)
        self.mix_lr_floor = float(mix_lr_floor)
        _nhalf = int(mix_lr_nhalf)
        self.mix_lr_nhalf = _nhalf
        self.mix_lr_inv = (1.0 / _nhalf) if _nhalf > 0 else 0.0
        if _nhalf > 0 and float(mix_lr_floor) >= 1.0:
            print("WARNING: mix_lr_nhalf>0 tapi mix_lr_floor>=1.0 -> faktor jadwal "
                  "terjepit ke 1 (jadwal NONAKTIF). Set floor<1 (mis. 0.02) bila "
                  "memang ingin menjadwalkan.")
        self.mix_l2 = float(mix_l2)
        self.cnt = np.zeros((3, 4096), dtype=np.int64)  # kunjungan per (set, baris)
        # #4: bucket asosiatif 4-way (opt[0]=1). Array `opt` = pola flag bertahap
        # agar tak menambah argumen kernel satu per satu. Default NOL = legacy.
        # opt[4] (fast_scan): DIPENSIUNKAN — pemindaian-tunggal terukur −0,065
        # (dirty-scan > rescan); update selalu pindai-ulang. opt[64:128] cadangan.
        self.opt = np.zeros(128, dtype=np.int64)
        self.opt[0] = 1 if assoc else 0
        self.opt[4] = 1 if fast_scan else 0
        self.opt[1] = 1 if hit_sel else 0
        self.drop_permille = int(dropout_permille)
        self.opt[9] = self.drop_permille
        self.opt[10] = 1 if match_adapt else 0
        self._mrun = np.zeros(1, dtype=np.int64)
        self.sm = np.full(32, 0.5, dtype=np.float64)
        self._rng = np.zeros(1, dtype=np.uint64)
        self._rng[0] = np.uint64(0x9E3779B97F4A7C15 ^ np.uint64(seed & 0xFFFFFF))
        if self.nctx > 32:
            raise ValueError(f"nctx={self.nctx} > 32 melebihi scratch vsl")
        # ★ fix kurva-U (2026-10-07): BEKUKAN mixer+APM setelah N byte latih (0 = tak pernah).
        # E6 mematahkan "magnitudo w1" sebagai sebab; E7 menunjukkan kerusakan dibawa MIXER
        # (cangkok mixer@3jt memulihkan 89%); E8: titik beku optimal ~3 jt byte. Karena update
        # TABEL independen dari mixer, menolkan lr/lr2/alr setelah N byte membekukan mixer+APM
        # secara EKSAK — setara cangkok `tab@N + mixer/APM@freeze_after` (diverifikasi E9).
        self.mixer_freeze_after = int(mixer_freeze_after)
        self._train_fed = 0
        nmix = self.nctx + 3                        # + match + neural + bias
        # Mixer 2 lapis (M3): lapis 1 = 3 mixer (pemilih c0 / prev_byte / (prev_byte,c0)),
        # lapis 2 = campuran ketiganya, dipilih oleh c0.
        self.w1 = np.full((3, 4096, nmix), 1.0 / self.nctx, dtype=np.float64)
        self.w2 = np.zeros((256, 4), dtype=np.float64)
        self.w2[:, :3] = 1.0 / 3.0
        self.lr2 = float(lr2)
        self.stab = build_stretch()
        self.use_apm = bool(use_apm)
        self.apm1 = build_apm(256)
        self.apm2 = build_apm(65536)
        self.alr = float(apm_lr)
        # model match (M2)
        self.match_minlen = int(match_minlen)
        self.match_conf = float(match_conf)
        msize = 1 << int(match_bits)
        self.mmask = msize - 1
        self.mchk = np.zeros(msize, dtype=np.uint8)
        self.mbyte = np.zeros(msize, dtype=np.uint8)
        self.hist = np.zeros(HBUF, dtype=np.uint8)
        self.pos = 0
        self.hlen = 0
        self.prev_byte = 0
        self.lr = float(lr)
        self.count_cap = int(count_cap)
        # cks_bits=32 meluapkan bit tanda int64 ((ck<<32) negatif -> tak pernah cocok).
        # Rentang sah 1..31 (audit eksternal; tak ada pemakai 32 — default 24).
        if not (1 <= int(cks_bits) <= 31):
            raise ValueError("cks_bits harus 1..31 (32 meluapkan bit tanda int64)")
        self.cks_bits = int(cks_bits)
        self.ck_mask = (1 << self.cks_bits) - 1
        # perbaikan match/ctx (bisa dimatikan untuk perbandingan terkontrol)
        self.match_fix = 1 if match_fix else 0
        self.ctx_fix = 1 if ctx_fix else 0
        if not (1 <= int(match_ck_bits) <= 8):
            raise ValueError("match_ck_bits harus 1..8 (mchk bertipe uint8)")
        self.mck_mask = (1 << int(match_ck_bits)) - 1
        self.n_bytes = 0
        self.ce_bits = 0.0
        # scratch
        self._diag = np.zeros(16, dtype=np.float64)
        self._rel = np.zeros(40, dtype=np.float64)
        self._keys = np.zeros(self.nctx, dtype=np.uint64)
        self._st = np.zeros(self.nctx + 3, dtype=np.float64)
        self._nn_row = np.zeros(8, dtype=np.float64)
        self._probs = np.zeros(8, dtype=np.float64)
        self._hctx = np.zeros(self.nctx, dtype=np.uint64)
        self._havail = np.zeros(self.nctx, dtype=np.uint8)

    def byte_probs(self, byte):
        """Probabilitas per bit (p(bit=1)) untuk byte ini, TANPA memperbarui apa pun.

        Menjalankan pass maju pada keadaan sekarang; tabel/penghitung/mixer TIDAK berubah
        (dipakai untuk tes paritas: produk probabilitas harus sama dengan 2^(-CE)).
        """
        byte = int(byte) & 255
        self._run(byte, train=False)
        return self._probs.copy()

    def _eff_rates(self):
        """Laju efektif (lr, lr2, alr) dengan mempertimbangkan titik beku mixer.

        Satu-satunya tempat yang menentukan laju (fix §11.2): semua jalur
        (`step`, `train_stream`, `instrument_stream`) memakai ini, sehingga
        `mixer_freeze_after` dihormati di mana pun. `self.lr/lr2/alr` selalu
        menyimpan laju BASIS (tak pernah dinolkan sementara).
        """
        if self.mixer_freeze_after > 0 and self._train_fed >= self.mixer_freeze_after:
            return 0.0, 0.0, 0.0
        return self.lr, self.lr2, self.alr

    def _run(self, byte, train, nn_st=None):
        use_nn = 0
        if nn_st is not None:
            self._nn_row[:] = nn_st
            use_nn = 1
        lr, lr2, alr = self._eff_rates()
        return stat_step(byte, self.hist, self.pos, self.hlen, self.tab,
                         self.mask, self.w1, self.w2, self.coff, self.clen, self.stab, lr,
                         lr2, 1 if train else 0, self._keys, self._st,
                         self.apm1, self.apm2, alr, self.prev_byte,
                         1 if self.use_apm else 0,
                         self.mchk, self.mbyte, self.mmask, self.match_minlen,
                         self.match_conf, self.wsel_mask, self._probs,
                         self._hctx, self._havail, self._nn_row, use_nn, self.count_cap,
                         self.ck_mask, self.match_fix, self.ctx_fix, self.mck_mask,
                         self.charclass, self.wsel_mode, self.stretch_exact, self.w_bound,
                         self._diag, 0, self._rel,
                         self.mixer_mode, self.s2_clip, self.mix_lr_floor,
                         self.mix_lr_inv, self.mix_l2, self.cnt, self.opt,
                         self._mrun, self.sm, self._rng)

    # ------------------------------------------------------------ inti
    def step(self, byte, train=True, nn_st=None):
        """Proses satu byte. Mengembalikan CE (bit) atau None untuk byte pertama.

        ``nn_st`` (opsional, panjang 8): masukan neural Tingkat B — ``stretch(p(bit|prefiks))``
        untuk kedelapan bit byte ini. Menambah satu masukan ke mixer; bobotnya terpelajar.
        """
        byte = int(byte) & 255
        if self.hlen == 0:                       # butuh riwayat minimal
            self._push(byte)
            self.prev_byte = byte
            if train:
                self._train_fed += 1
            return None
        ce = self._run(byte, train, nn_st)
        self._push(byte)
        self.prev_byte = byte
        self.n_bytes += 1
        self.ce_bits += ce
        if train:
            self._train_fed += 1
        return ce

    def _push(self, byte):
        self.hist[self.pos & (HBUF - 1)] = byte
        self.pos += 1
        if self.hlen < HBUF:
            self.hlen += 1

    # ------------------------------------------------------------ pemakaian
    def eval_stream(self, data, nn_stream=None):
        """CE prekuensial per byte TANPA belajar (tabel beku, konteks tetap berjalan).

        ``nn_stream`` (opsional, (N,8)): masukan neural per byte untuk seluruh aliran.
        """
        out = []
        for i, b in enumerate(data):
            row = None if nn_stream is None else nn_stream[i]
            c = self.step(b, train=False, nn_st=row)
            if c is not None:
                out.append(c)
        return np.asarray(out, dtype=np.float64)

    # ------------------------------------------------------------ jalur CEPAT (numba)
    def _nn_prep(self, data, nn_stream):
        """Validasi ``nn_stream``; kembalikan ``(array | None, use_nn)``.

        Bentuk wajib ``(len(data), 8)`` — 8 stretch (log-odds) per byte, format yang sama
        dengan ``nn_st`` pada ``step()``. Sumber nyata: `tmt2/byte_bridge.py` (LLM token-level
        -> byte-level).
        """
        if nn_stream is None:
            return None, 0
        ns = np.ascontiguousarray(nn_stream, dtype=np.float64)
        if ns.shape != (len(data), 8):
            raise ValueError(f"nn_stream harus berbentuk ({len(data)}, 8), dapat {ns.shape}")
        return ns, 1

    def _stream(self, data, train, out=None, nn_stream=None):
        """Satu panggilan numba untuk seluruh aliran; memutakhirkan state internal.

        Mengembalikan CE (bit) per byte, entri pertama ``nan`` (byte tanpa riwayat).
        ``out`` (opsional, int64[3]) menerima [pos, hlen, prev_byte] hasil akhir.
        ``nn_stream`` (opsional, (N,8)) = masukan guru per byte.
        """
        data = np.ascontiguousarray(data, dtype=np.uint8)
        ns, use_nn = self._nn_prep(data, nn_stream)
        lr, lr2, alr = self._eff_rates()
        return stat_stream(data, self.hist, self.pos, self.hlen, self.tab,
                           self.mask, self.w1, self.w2, self.coff, self.clen, self.stab,
                           lr, lr2, 1 if train else 0, self._keys, self._st,
                           self.apm1, self.apm2, alr, self.prev_byte,
                           1 if self.use_apm else 0,
                           self.mchk, self.mbyte, self.mmask, self.match_minlen,
                           self.match_conf, self.wsel_mask, self._probs,
                           self._hctx, self._havail, self._nn_row, use_nn, self.count_cap,
                           self.ck_mask, self.match_fix, self.ctx_fix, self.mck_mask,
                           self.charclass, self.wsel_mode, self.stretch_exact, out,
                           self.w_bound, self.mixer_mode, self.s2_clip,
                           self.mix_lr_floor, self.mix_lr_inv, self.mix_l2, self.cnt,
                           self.opt, self._mrun, self.sm, self._rng, ns)

    def _train_stream_inner(self, data, nn_stream=None):
        out = np.zeros(3, dtype=np.int64)
        ce = self._stream(data, train=True, out=out, nn_stream=nn_stream)
        self.pos, self.hlen, self.prev_byte = int(out[0]), int(out[1]), int(out[2])
        self.n_bytes += int(np.count_nonzero(~np.isnan(ce)))
        self.ce_bits += float(np.nansum(ce))
        return ce

    def train_stream(self, data, nn_stream=None):
        """Latih SELURUH ``data`` dalam satu panggilan terkompilasi (tanpa loop Python).

        Ini jalur produksi cepat: menghapus overhead pemanggilan Python per byte yang
        sebelumnya membatasi ke ~90 kB/s. Kembalian CE (bit) per byte (``nan`` di awal).

        ``mixer_freeze_after`` (bila > 0) membekukan mixer+APM setelah sekian byte latih:
        tabel tetap belajar, tetapi laju efektif (`_eff_rates()`) menjadi nol sehingga
        bobot mixer & APM tak lagi berubah (fix kurva-U; lihat HASIL_E7_PEMISAHAN.md /
        HASIL_E8). Aliran dipecah tepat di titik beku agar tiap chunk punya laju konstan.

        ``nn_stream`` (opsional, (N,8)) = masukan guru per byte (mis. dari `tmt2/byte_bridge.py`);
        dipecah bersamaan dengan ``data`` bila ``mixer_freeze_after`` aktif.

        ⚠️ `_train_fed` dinaikkan SETELAH chunk dilatih (bukan sebelumnya) supaya laju
        efektif dihitung dari keadaan AWAL chunk — sama seperti `step()`. Versi lama
        menaikkannya lebih dulu, sehingga chunk yang tepat berakhir di titik beku ikut
        membeku (bug batas; ditangkap `test_nn_stream_sliced_with_freeze_split`).
        """
        fa = self.mixer_freeze_after
        n = len(data)
        if fa <= 0:
            ce = self._train_stream_inner(data, nn_stream)
            self._train_fed += n
            return ce
        remaining = fa - self._train_fed
        if remaining <= 0:                          # sudah lewat titik beku -> laju 0
            ce = self._train_stream_inner(data, nn_stream)
            self._train_fed += n
            return ce
        if remaining >= n:                          # seluruh chunk masih bebas
            ce = self._train_stream_inner(data, nn_stream)
            self._train_fed += n
            return ce
        ns1 = None if nn_stream is None else nn_stream[:remaining]
        ns2 = None if nn_stream is None else nn_stream[remaining:n]
        ce1 = self._train_stream_inner(data[:remaining], ns1)   # fase bebas
        self._train_fed += remaining               # == fa -> fase berikut beku
        ce2 = self._train_stream_inner(data[remaining:], ns2)   # fase beku
        self._train_fed += n - remaining
        return np.concatenate([ce1, ce2])

    def eval_stream_fast(self, data, nn_stream=None):
        """Seperti ``eval_stream`` tetapi dalam satu panggilan numba (CE bit, ``nan`` awal)."""
        out = np.zeros(3, dtype=np.int64)
        ce = self._stream(data, train=False, out=out, nn_stream=nn_stream)
        self.pos, self.hlen, self.prev_byte = int(out[0]), int(out[1]), int(out[2])
        self.n_bytes += int(np.count_nonzero(~np.isnan(ce)))
        return ce

    # ------------------------------------------------------------ diagnostik Fase 0
    def stream_ctx(self):
        """Snapshot konteks stream untuk save/restore.

        Tuple (hist, pos, hlen, prev, mrun, sm, rng) — 4 entri pertama STABIL
        (kode lama memakai snap[0]/snap[1] posisional); 3 terakhir ditambahkan
        untuk state adaptif (C2-L1 + dropout RNG).
        """
        return (self.hist.copy(), self.pos, self.hlen, self.prev_byte,
                int(self._mrun[0]), self.sm.copy(), int(self._rng[0]))

    def restore_ctx(self, snap):
        if len(snap) == 4:  # snapshot lama (pra-C2): state adaptif tak tersentuh
            h, pos, hlen, prev = snap
        else:
            h, pos, hlen, prev, mrun, sm, rng = snap
            self._mrun[0] = mrun
            self.sm[:] = sm
            self._rng[0] = rng
        self.hist[:] = h
        self.pos, self.hlen, self.prev_byte = pos, hlen, prev

    def mix_snapshot(self):
        """Snapshot MURAH mixer+APM (~20 MB) untuk cangkok E7b (tanpa tabel)."""
        return dict(w1=self.w1.copy(), w2=self.w2.copy(),
                    apm1=self.apm1.copy(), apm2=self.apm2.copy())

    def load_mix(self, snap):
        self.w1[:] = snap["w1"]
        self.w2[:] = snap["w2"]
        self.apm1[:] = snap["apm1"]
        self.apm2[:] = snap["apm2"]

    def tab_snapshot(self):
        return self.tab.copy()

    def load_tab(self, tab):
        self.tab[:] = tab

    def eval_suite(self, windows, cold=True):
        """CE rata-rata per jendela TANPA menyalin tabel (hemat 134-268 MB/jendela).

        Tiap jendela dinilai dgn konteks stream dingin-beku bila ``cold`` (hist/pos
        di-nol-kan lalu dipulihkan), atau konteks berjalan bila False. Tabel, mixer,
        APM, match TIDAK berubah (train=False) — dijaga tes.
        Mengembalikan (mean, per_list).
        """
        ctx = self.stream_ctx()
        per = []
        for w in windows:
            if cold:
                self.hist[:] = 0
                self.pos, self.hlen, self.prev_byte = 0, 0, 0
            else:
                self.restore_ctx(ctx)
            ce = self.eval_stream_fast(np.ascontiguousarray(w, dtype=np.uint8))
            per.append(float(np.nanmean(ce)))
        self.restore_ctx(ctx)
        return float(np.mean(per)), per

    def diagnose(self, data):
        """Jalankan ``data`` (train=False, tanpa mutasi) + kembalikan diagnostik Fase 0.

        Mengembalikan dict: nbit, sat_frac (3), own_ce (3), pm_ce, pf_ce (=CE),
        rel (20 bin: count, freq empiris), w2neg_frac, argmax w1 (k, sel, c, val).
        Konteks stream di-restore setelahnya.
        """
        data = np.ascontiguousarray(data, dtype=np.uint8)
        ctx = self.stream_ctx()
        diag = np.zeros(16, dtype=np.float64)
        rel = np.zeros(40, dtype=np.float64)
        lr, lr2, alr = self._eff_rates()
        ce = stat_diagnose(data, self.hist, self.pos, self.hlen, self.tab,
                           self.mask, self.w1, self.w2, self.coff, self.clen,
                           self.stab, lr, lr2, self._keys, self._st,
                           self.apm1, self.apm2, alr, self.prev_byte,
                           1 if self.use_apm else 0,
                           self.mchk, self.mbyte, self.mmask, self.match_minlen,
                           self.match_conf, self.wsel_mask, self._probs,
                           self._hctx, self._havail, self._nn_row, 0, self.count_cap,
                           self.ck_mask, self.match_fix, self.ctx_fix, self.mck_mask,
                           self.charclass, self.wsel_mode, self.stretch_exact,
                           self.w_bound, diag, rel,
                           self.mixer_mode, self.s2_clip, self.mix_lr_floor,
                           self.mix_lr_inv, self.mix_l2, self.cnt, self.opt,
                           self._mrun, self.sm, self._rng)
        self.restore_ctx(ctx)
        nbit = max(diag[0], 1.0)
        out = dict(
            nbit=float(diag[0]), ce=float(ce),
            sat_frac=[float(diag[1 + k] / nbit) for k in range(3)],
            own_ce=[float(diag[4 + k] / nbit) for k in range(3)],
            pm_ce=float(diag[7] / nbit), pf_ce=float(diag[8] / nbit),
            rel_count=rel[:20].copy(),
            rel_freq=[float(rel[20 + b] / rel[b]) if rel[b] > 0 else float("nan")
                      for b in range(20)],
            w2neg_frac=float(np.mean(self.w2[:, :3] < 0.0)),
        )
        ax = np.unravel_index(int(np.argmax(np.abs(self.w1))), self.w1.shape)
        out["w1argmax"] = dict(k=int(ax[0]), sel=int(ax[1]), c=int(ax[2]),
                               val=float(self.w1[ax]))
        out["w1max"] = float(np.max(np.abs(self.w1)))
        out["w1mean"] = float(np.mean(np.abs(self.w1)))
        return out

    def instrument_stream(self, data, interval, probe_bytes=20000):
        """E2: latih ``data`` dengan probe internal per ``interval`` byte (SEMUA terkompilasi).

        Mengembalikan list catatan seperti ``scripts/e2_instrument.py`` tetapi tanpa pajak
        loop Python. Probe dijalankan ``train=False`` dan posisi stream dipulihkan; isi
        tabel/mixer/APM TIDAK berubah → hasil latih identik.
        """
        data = np.ascontiguousarray(data, dtype=np.uint8)
        fa = self.mixer_freeze_after
        n = len(data)
        if fa <= 0 or self._train_fed >= fa or self._train_fed + n <= fa:
            # satu rezim laju -> satu panggilan (laju efektif via _eff_rates)
            self._train_fed += n
            return self._instrument_inner(data, interval, probe_bytes, off=0)
        # pecah tepat di titik beku (tiap bagian satu rezim laju). Catatan: jadwal
        # probe interval dimulai ulang di bagian kedua; probe tak mengubah hasil latih
        # (train=False + restore), hanya posisi sampling diagnostik yang bergeser.
        remaining = fa - self._train_fed
        h1 = self._instrument_inner(data[:remaining], interval, probe_bytes, off=0)
        self._train_fed += remaining              # == fa -> bagian berikut beku
        h2 = self._instrument_inner(data[remaining:], interval, probe_bytes,
                                    off=remaining)
        self._train_fed += n - remaining
        return tuple(np.concatenate([x, y]) for x, y in zip(h1, h2))

    def _instrument_inner(self, data, interval, probe_bytes, off):
        ncap = len(data) // int(interval) + 1
        out_hr = np.zeros((ncap, self.nctx), dtype=np.float64)
        out_av = np.zeros((ncap, self.nctx), dtype=np.float64)
        out_idx = np.zeros(ncap, dtype=np.int64)
        out_w1mx = np.zeros(ncap, dtype=np.float64)
        out_w1mn = np.zeros(ncap, dtype=np.float64)
        out_w2mn = np.zeros(ncap, dtype=np.float64)
        out_fill = np.zeros(ncap, dtype=np.float64)
        out = np.zeros(3, dtype=np.int64)
        lr, lr2, alr = self._eff_rates()
        k = stat_instrument(data, int(interval), int(probe_bytes), self.hist, self.pos,
                            self.hlen, self.tab, self.mask, self.w1, self.w2, self.coff,
                            self.clen, self.stab, lr, lr2, self._keys, self._st,
                            self.apm1, self.apm2, alr, self.prev_byte,
                            1 if self.use_apm else 0, self.mchk, self.mbyte, self.mmask,
                            self.match_minlen, self.match_conf, self.wsel_mask, self._probs,
                            self._hctx, self._havail, self._nn_row, 0, self.count_cap,
                            self.ck_mask, self.match_fix, self.ctx_fix, self.mck_mask,
                            self.charclass, self.wsel_mode, self.stretch_exact,
                            out_hr, out_av, out_idx, out_w1mx, out_w1mn, out_w2mn,
                            out_fill, out, self.w_bound, self.mixer_mode, self.s2_clip,
                            self.mix_lr_floor, self.mix_lr_inv, self.mix_l2, self.cnt,
                            self.opt, self._mrun, self.sm, self._rng)
        self.pos, self.hlen, self.prev_byte = int(out[0]), int(out[1]), int(out[2])
        return (out_hr[:k], out_av[:k], out_idx[:k] + off,
                out_w1mx[:k], out_w1mn[:k], out_w2mn[:k], out_fill[:k])

    def bpb(self, data):
        ce = self.eval_stream(data)
        return float(ce.mean()) if len(ce) else float("nan")

    # ------------------------------------------------------------ generasi
    def _sample_byte(self, rng, temp=1.0, min_p=0.0):
        """Sampel satu byte dari model pada keadaan sekarang, TANPA memperbarui apa pun.
        `min_p>0`: nolkan byte berprobabilitas < min_p*pmax lalu renormalisasi (#2).

        Model sudah dekomposisi biner: tiap bit adalah keputusan bernoulli. Untuk byte b,
        ``byte_probs(b)`` memberi p(bit=1) untuk KEDELAPAN bit; probabilitas byte =
        produk p(bit = nilai bit SEBENARNYA):

            p(b) = ∏_i  [ pb_i        jika bit_i(b) == 1
                          1 - pb_i     jika bit_i(b) == 0 ]

        ⚠️ JEBAKAN (bug ditemukan 2026-10-04): versi pertama memakai ``np.prod(pb)`` — yaitu
        mengalikan p(bit=1) untuk SEMUA bit seolah tiap bit bernilai 1. Itu membuat byte
        dengan banyak bit-0 (teks ASCII) dihukum berat dan byte biner (bit-1 di posisi tinggi)
        menang → generasi keluar byte sampah >127. Harus memakai nilai bit byte itu.
        ``temp`` menskalakan LOGIT (1 = apa adanya).
        """
        probs = np.empty(256, dtype=np.float64)
        for b in range(256):
            pb = np.clip(self.byte_probs(b), 1e-12, 1 - 1e-12)
            if temp != 1.0:
                # skala logit: p' = squash(logit(p)/temp), logit = ln(p/(1-p))
                lg = np.log(pb / (1 - pb)) / temp
                pb = 1.0 / (1.0 + np.exp(-lg))
            # kalikan menurut NILAI bit byte b, bukan p(bit=1) mentah
            lik = np.where(((b >> np.arange(7, -1, -1)) & 1).astype(bool), pb, 1.0 - pb)
            probs[b] = float(np.prod(lik))
        if min_p > 0.0:
            probs[probs < float(min_p) * probs.max()] = 0.0
        tot = probs.sum()
        if tot <= 0 or not np.isfinite(tot):
            probs[:] = 1.0 / 256.0
        else:
            probs /= tot
        return int(rng.choice(256, p=probs))

    def generate(self, prompt, n, temp=0.8, seed=0, ancestral=False, min_p=0.01):
        # Default min_p=0,01 dari sweep G0b (M5≈riil 0,28, M6 minimal; lihat HASIL_E7B_V3).
        """Hasilkan ``n`` byte lanjutan dari ``prompt`` (byte/str). TANPA belajar.

        Prompt memberi konteks (diumpankan ke keadaan), lalu tiap byte berikutnya disampel
        dan UMPAN-BALIK sebagai konteks — persis protokol generasi autoregresif. Model TIDAK
        diperbarui selama generasi (meniru mode inferensi beku).

        Konteks stream DIRESTORE setelah selesai (fix T0.2): dua panggilan berurutan
        dengan seed sama memberi hasil identik; metrik lintas-sampel tak tercemar
        urutan pemanggilan. ``ancestral=True`` memakai sampler 8-pass (~32× lebih
        cepat, distribusi identik pada temp=1).
        """
        if isinstance(prompt, str):
            prompt = prompt.encode("utf-8")
        rng = np.random.default_rng(seed)
        ctx, nbc = self.stream_ctx(), (self.n_bytes, self.ce_bits)
        try:
            out = bytearray()
            for b in bytes(prompt):
                self.step(int(b), train=False)   # isi konteks, tanpa belajar
            for _ in range(int(n)):
                if ancestral:
                    if min_p > 0.0:
                        raise ValueError("min-p butuh distribusi penuh: pakai ancestral=False")
                    b = self._sample_byte_ancestral(rng, temp=temp)
                else:
                    b = self._sample_byte(rng, temp=temp, min_p=min_p)
                out.append(b)
                self.step(b, train=False)        # umpankan byte yang dihasilkan
            return bytes(out)
        finally:
            self.restore_ctx(ctx)
            self.n_bytes, self.ce_bits = nbc

    def _bit_prob_prefix(self, prefix, i, temp=1.0):
        """p(bit_i=1 | prefiks tersampel) via byte-dummy berprefiks sama.

        Kausalitas kernel: probs[i] hanya bergantung bit 0..i-1 (c0 prefiks +
        pemutusan mpred prefiks), jadi bit dummy setelah i tak berpengaruh.
        """
        dummy = (prefix << (8 - i)) & 255
        pb = np.clip(self.byte_probs(dummy), 1e-12, 1 - 1e-12)
        p = float(pb[i])
        if temp != 1.0:
            lg = np.log(p / (1.0 - p)) / temp
            p = 1.0 / (1.0 + np.exp(-lg))
        return p

    def _sample_byte_ancestral(self, rng, temp=1.0):
        """Sampler ancestral bit-demi-bit: 8 panggilan kernel per byte (vs 256).

        Distribusi identik dengan `_sample_byte` pada temp=1 (aturan rantai).
        Untuk min-p/top-p tetap pakai `_sample_byte` (butuh distribusi penuh).
        """
        prefix = 0
        for i in range(8):
            p = self._bit_prob_prefix(prefix, i, temp=temp)
            bit = 1 if rng.random() < p else 0
            prefix = (prefix << 1) | bit
        return prefix

    @classmethod
    def from_config(cls, cfg):
        """Bangun StatCore dari `Config` (preset `Config.v2()`).

        ⚠️ B-06: `Config` TIDAK punya field untuk cks_bits/count_cap/match_fix/ctx_fix/
        match_ck_bits, sehingga jalur ini memakai DEFAULT konstruktor untuk field itu.
        Bila Config punya field `stat_cks_bits` dst (versi baru), field itu diteruskan;
        kalau tidak, default dipakai dan dicatat di `self._cfg_source`.
        """
        orders = tuple(int(x) for x in str(cfg.stat_orders).split(",") if x.strip() != "")
        opt = {}
        # Teruskan SEMUA knob (§11.4). getattr dgn default = kompatibel dgn objek
        # Config lama / duck-type yang belum punya field baru.
        for fld, param in (("stat_cks_bits", "cks_bits"), ("stat_count_cap", "count_cap"),
                           ("stat_match_fix", "match_fix"), ("stat_ctx_fix", "ctx_fix"),
                           ("stat_match_ck_bits", "match_ck_bits"),
                           ("stat_wsel_mode", "wsel_mode"),
                           ("stat_stretch_exact", "stretch_exact"),
                           ("stat_w_bound", "w_bound"),
                           ("stat_mixer_freeze_after", "mixer_freeze_after"),
                           ("stat_mixer_mode", "mixer_mode"),
                           ("stat_s2_clip", "s2_clip"),
                           ("stat_mix_lr_floor", "mix_lr_floor"),
                           ("stat_mix_lr_nhalf", "mix_lr_nhalf"),
                           ("stat_mix_l2", "mix_l2"),
                           ("stat_assoc", "assoc"),
                           ("stat_fast_scan", "fast_scan"),
                           ("stat_hit_sel", "hit_sel"),
                           ("stat_dropout_permille", "dropout_permille"),
                           ("stat_match_adapt", "match_adapt")):
            if hasattr(cfg, fld):
                opt[param] = getattr(cfg, fld)
        lr2 = float(getattr(cfg, "stat_lr2", 0.003))  # default = default StatCore
        return cls(orders=orders, table_bits=int(cfg.stat_hash_bits), lr=float(cfg.stat_lr),
                   use_apm=bool(cfg.stat_apm), apm_lr=float(cfg.stat_apm_lr),
                   match_minlen=int(cfg.stat_match_minlen), match_bits=int(cfg.stat_match_bits),
                   match_conf=float(cfg.stat_match_conf), wsel_bits=int(cfg.stat_wsel_bits),
                   lr2=lr2, sparse=bool(cfg.stat_sparse), word=bool(cfg.stat_word),
                   **opt)

    def hyper(self):
        """Kembalikan SEMUA hiperparameter konstruktor (untuk checkpoint yang lengkap, B-06)."""
        return dict(orders=[int(x) for x in self.orders], table_bits=int(self.mask.bit_length()),
                    lr=float(self.lr), use_apm=bool(self.use_apm), apm_lr=float(self.alr),
                    match_minlen=int(self.match_minlen), match_bits=int((self.mmask + 1).bit_length() - 1),
                    match_conf=float(self.match_conf), wsel_bits=int(self.wsel_bits),
                    lr2=float(self.lr2), count_cap=int(self.count_cap), cks_bits=int(self.cks_bits),
                    match_fix=bool(self.match_fix), ctx_fix=bool(self.ctx_fix),
                    match_ck_bits=int(self.mck_mask.bit_length()),
                    sparse=bool(getattr(self, "_sparse", False)),
                    word=bool(getattr(self, "_word", False)),
                    wsel_mode=int(self.wsel_mode), stretch_exact=int(self.stretch_exact),
                    w_bound=float(self.w_bound),
                    mixer_freeze_after=int(self.mixer_freeze_after),
                    mixer_mode=int(self.mixer_mode), s2_clip=float(self.s2_clip),
                    mix_lr_floor=float(self.mix_lr_floor),
                    mix_lr_nhalf=int(self.mix_lr_nhalf), mix_l2=float(self.mix_l2),
                    assoc=int(self.opt[0]), fast_scan=int(self.opt[4]),
                    hit_sel=int(self.opt[1]),
                    dropout_permille=int(self.drop_permille),
                    match_adapt=int(self.opt[10]))

    @classmethod
    def from_hyper(cls, h, seed=0):
        """Bangun StatCore dari dict `hyper()` — kebalikan yang bit-identik (tes round-trip)."""
        return cls(orders=tuple(h["orders"]), table_bits=int(h["table_bits"]), lr=float(h["lr"]),
                   seed=seed, use_apm=bool(h["use_apm"]), apm_lr=float(h["apm_lr"]),
                   match_minlen=int(h["match_minlen"]), match_bits=int(h["match_bits"]),
                   match_conf=float(h["match_conf"]), wsel_bits=int(h["wsel_bits"]),
                   lr2=float(h["lr2"]), sparse=bool(h.get("sparse", True)),
                   word=bool(h.get("word", True)), count_cap=int(h["count_cap"]),
                   cks_bits=int(h["cks_bits"]), match_fix=bool(h["match_fix"]),
                   ctx_fix=bool(h["ctx_fix"]), match_ck_bits=int(h["match_ck_bits"]),
                   wsel_mode=int(h.get("wsel_mode", 0)),
                   stretch_exact=int(h.get("stretch_exact", 0)),
                   w_bound=float(h.get("w_bound", 0.0)),
                   mixer_freeze_after=int(h.get("mixer_freeze_after", 0)),
                   assoc=int(h.get("assoc", 0)),
                   fast_scan=int(h.get("fast_scan", 0)),
                   hit_sel=int(h.get("hit_sel", 0)),
                   dropout_permille=int(h.get("dropout_permille", 0)),
                   match_adapt=int(h.get("match_adapt", 0)),
                   mixer_mode=int(h.get("mixer_mode", 0)),
                   s2_clip=float(h.get("s2_clip", 16.0)),
                   mix_lr_floor=float(h.get("mix_lr_floor", 1.0)),
                   mix_lr_nhalf=int(h.get("mix_lr_nhalf", 0)),
                   mix_l2=float(h.get("mix_l2", 0.0)))

    CKPT_VERSION = 4
    # v1 = format awal (tanpa train_fed) — lihat train_gen_big.save_ckpt lama.
    # v2 = + train_fed (posisi titik beku; fix §11.3).
    # v3 = + cnt (hitungan kunjungan mixer; Mixer v3 Fase 1).
    # v4 = + mrun/sm/rngstate (state adaptif C2-L1 + dropout).

    def save(self, path, meta=None):
        """Simpan checkpoint lengkap: array state + train_fed + versi + meta.

        Sumber kebenaran tunggal untuk format checkpoint (§11.3). Semua skrip
        harus memakai ini (bukan np.savez_compressed manual) agar field baru
        tak tercecer lagi.
        """
        import os
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        import json
        meta = dict(meta or {})
        meta.setdefault("hyper", self.hyper())
        np.savez_compressed(path, tab=self.tab, w1=self.w1, w2=self.w2,
                            apm1=self.apm1, apm2=self.apm2,
                            mchk=self.mchk, mbyte=self.mbyte, hist=self.hist,
                            pos=np.int64(self.pos), hlen=np.int64(self.hlen),
                            prev_byte=np.int64(self.prev_byte),
                            train_fed=np.int64(self._train_fed),
                            cnt=self.cnt,
                            mrun=np.int64(self._mrun[0]), sm=self.sm,
                            rngstate=np.uint64(self._rng[0]),
                            ckpt_version=np.int64(self.CKPT_VERSION),
                            meta=np.array([json.dumps(meta)], dtype=object))

    @classmethod
    def load(cls, path):
        """Muat checkpoint yang disimpan `save()`. Mengembalikan (model, meta).

        Kompatibel mundur: checkpoint v1 (tanpa train_fed) dimuat dengan
        `_train_fed=0` + peringatan bila `mixer_freeze_after>0` (melanjutkan latih
        dari ckpt lama akan membuka mixer selama K byte — catat di log run).
        """
        d = np.load(path, allow_pickle=True)
        import json
        meta = json.loads(str(d["meta"][0]))
        if "hyper" in meta:
            m = cls.from_hyper(meta["hyper"])
        else:  # ckpt sangat lama (pra-B-06)
            print("WARNING: checkpoint tanpa 'hyper' lengkap -> default dipakai")
            m = cls(orders=tuple(meta["orders"]), table_bits=int(meta["table_bits"]),
                    lr=float(meta["lr"]), use_apm=bool(meta["use_apm"]),
                    wsel_bits=int(meta["wsel_bits"]), sparse=bool(meta["sparse"]),
                    word=True, cks_bits=int(meta["cks_bits"]))
        m.tab[:] = d["tab"]
        m.w1[:] = d["w1"]
        m.w2[:] = d["w2"]
        m.apm1[:] = d["apm1"]
        m.apm2[:] = d["apm2"]
        m.mchk[:] = d["mchk"]
        m.mbyte[:] = d["mbyte"]
        m.hist[:] = d["hist"]
        m.pos = int(d["pos"])
        m.hlen = int(d["hlen"])
        m.prev_byte = int(d["prev_byte"])
        if "train_fed" in d:
            m._train_fed = int(d["train_fed"])
        else:
            m._train_fed = 0
            if m.mixer_freeze_after > 0:
                print("WARNING: ckpt v1 tanpa train_fed + freeze>0 -> _train_fed=0; "
                      "lanjut latih akan membuka mixer selama freeze_after byte")
        if "cnt" in d:
            m.cnt[:] = d["cnt"]
        else:
            m.cnt[:] = 0
            if m.mixer_mode == 1 and m.mix_lr_inv > 0:
                print("WARNING: ckpt pra-v3 tanpa cnt + jadwal aktif -> cnt=0; "
                      "jadwal dimulai dari awal")
        if "mrun" in d:
            m._mrun[0] = int(d["mrun"])
            m.sm[:] = d["sm"]
            m._rng[0] = np.uint64(d["rngstate"])
        else:
            m._mrun[0] = 0
            m.sm[:] = 0.5
            # rng: seed ulang dari 0 (divergen dari run asli bila dropout aktif;
            # ckpt pra-v4 + dropout = kombinasi tak-valid — beri peringatan)
            if m.drop_permille > 0:
                print("WARNING: ckpt pra-v4 tanpa rngstate + dropout aktif -> "
                      "RNG di-seed ulang (divergen dari run asli)")
        return m, meta

    def reset(self):
        self.tab[:] = 0
        self.cnt[:] = 0
        self._mrun[0] = 0
        self.sm[:] = 0.5
        self.w1[:] = 1.0 / self.nctx
        self.w2[:] = 0.0
        self.w2[:, :3] = 1.0 / 3.0
        self.apm1[:] = build_apm(256)
        self.apm2[:] = build_apm(65536)
        self.mchk[:] = 0
        self.mbyte[:] = 0
        self.hist[:] = 0
        self.pos = 0
        self.hlen = 0
        self.prev_byte = 0
        self.n_bytes = 0
        self.ce_bits = 0.0
        self._train_fed = 0
