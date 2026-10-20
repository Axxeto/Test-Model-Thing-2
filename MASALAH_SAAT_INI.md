# MASALAH SAAT INI — TMT2 v2 (2026-10-10, revisi lengkap)

> Status tes: **124 lolos**. Dokumen konsolidasi: `DOKUMENTASI_LENGKAP.md`.
> Hasil terukur terbaru: `HASIL_E7B_V3.md` · `STATUS.md` §21 · `PLAN_V3.md`.
> Semua angka di bawah terukur di mesin ini (numba 0.68.0), bukan perkiraan.

---

## 0. Konteks satu paragraf

TMT2 v2 = model bahasa byte-stream online (prediksi byte ke-t sebelum melihatnya,
metrik BPB), numpy+numba, CPU-only. Jalur produksi = **Tingkat A**
(`tmt2/stat_core.py`): tabel hash + Mixer-2 + APM. Sejak 2026-10-10 default =
**Mixer v3 galat-sendiri polos (V2)** — kurva-U katastrofik (+0,34) hilang di
semua run 30–186M (datar ±0,01, max|w1| ~1–3). Tingkat B (neural) gagal G2,
modul riset. Model terbaik: 186M byte Indonesia, **BPB 2,0254** (tb26, V2);
dengan 4-way @30M **1,8352**.

---

## 1. Memori (RAM + checkpoint + state)

| komponen | ukuran | catatan |
|---|---|---|
| tabel tb24/25/26/27 | 134 / 268 / 536 / 1072 MB | 1×`int64`/slot; tb27 1GB masih full-speed |
| mixer+APM+match | ~20 MB | murah disnapshot (`mix_snapshot`) untuk cangkok |
| `checkpoints/gate_v2_tb26_186m.npz` | ~600MB (estimasi) | ckpt 30M = 293MB; format v3 (`train_fed`+`cnt`) |
| `checkpoints/v3_valid_5m.npz` | ~150MB | putaran validasi, reload bit-identik |
| konteks stream | 64 B hist + 3 int | wajib snapshot/restore di eval & generasi |

Masalah: (a) tabel SELALU jenuh 100% (horizon efektif 2–7M byte di tb26);
(b) M4 `gen_health` tak bisa bangun hash 24-gram prefix-186M (1,5GB) → dipakai
`bytes.find` per-sampel; (c) run paralel tb27+ butuh cek RAM bebas (~6,5GB/16GB).

## 2. Arsitektur

* **Tingkat A (produksi):** dekomposisi biner 8-bit, 10–17 konteks (dense +
  kata + jarang-opsional + match + neural-opsional), Mixer-2 (3 set terpilih
  konteks) + APM 2 tahap. Aturan belajar lapis-1 = **galat-sendiri per set**
  (V2, konveks per baris); lapis-2 + APM = galat akhir (tetap).
* **Legacy (pin):** mode `mixer_mode=0` + `Config.best()` simbolik dipertahankan
  bit-identik (`test_golden_legacy_values`) — bukan default, tapi acuan regresi.
* **Tingkat B (riset, G2 gagal):** Saron+Gong+gerbang+SwiGLU+RMSNorm, d=32,
  BPTT per-chunk numpy (~1–3kB/s). Paritas matematis lolos; kualitas Δ −0,0005.
  Syarat buka-kembali: backend ber-batch ≥30kB/s + Δ≥+0,05 (lihat §7).
* **Utang desain:** `w2`/APM masih LR-konstan; input = stretch KT mentah
  (tanpa state-map); `wsel_mode=1` belum diuji skala besar; C2-L1 (match
  sadar-panjang) menunggu array `opt`.

## 3. Kecepatan (terukur, satu proses)

| jalur | B/s | status |
|---|---|---|
| Tier A latih tb22–25 | 130–190k | ✅ ≥100k (klaim lulus) |
| Tier A eval | ~210k | ✅ |
| V3 vs legacy | −2,6% (A/B 180,8k→176,1k) | ✅ negligible |
| 4-way vs plain | **0,73×** (101k→74k) | 🔴 gagal ambang 0,92× → optimasi atau tb27 |
| Legacy SSM fused | 8k (mixer mati) / 2,7k (produksi) | arsip, bukan jalur |
| Tier B latih | 1–3k | 🔴 butuh BLAS/GPU sebelum riset lanjut |
| Sampler 256-pass → ancestral 8-pass | 32× (96 sampel 500B: 216s→16s) | ✅ |

Catatan: mesin bersama → ayunan ±30–50% antar jam; angka A/B selalu selang-seling.

## 4. Kualitas BPB (angka yang boleh dikutip)

| protokol | BPB |
|---|---|
| WikiText 200k/8k (V2-default) | ~1,79 (legacy 1,7846; V2 +0,007) |
| Indonesia 200k/8k | ~2,34 (V2 +0,008 vs legacy) |
| Indonesia 30M tb25, V2 | 2,1189 (legacy 2,1929 / sparse-ON 2,4650) |
| Indonesia 30M tb26, V2 | 2,0317 (legacy 2,2090) |
| Indonesia 30M tb26, V2+4-way | **1,8352** (Δ −0,197) |
| Indonesia 30M tb27, V2 | **1,9503** (tanpa kode) |
| Indonesia 186M tb26, V2 | **2,0254**, datar 60M→186M ±0,003 |
| Baseline adil 30M | lzma 2,2680 · bz2 2,4572 · zlib 3,2888 (semua kalah) |

Masalah kualitas tersisa: (a) premi V2 ≤1M −0,008 (1 seed, butuh multi-offset);
(b) benjolan 18M +0,011 transien-data (ambang, hilang saat diacak);
(c) kurva V3 datar-sejak-6M → gain 190M ≈ ±0,02 (kapasitas, bukan belajar).

## 5. Kualitas generasi (metrik M1–M7, ckpt 186M)

| metrik | T=1,0 | min-p 0,05 | teks riil |
|---|---|---|---|
| M1 LCS-silang (kolaps?) | 250 = kolaps penuh | 250 | 178 |
| M2 LCS-seed | 107 | 119 | — |
| M4 mem-24 / copy | 0,010 / 24 | 0,030 / 20 | 0,064 |
| M5 non-kata | 0,396 | **0,141** | 0,270 |
| M6 kalibrasi | 0,369 | 0,549 | — |
| M7 distinct-4 | 0,925 | 0,858 | 0,893 |

Bacaan: kolaps **distribusional** (min-p tak mengubah M1); menyalin BUKAN masalah
(model < riil); min-p memperbaiki kata-valid −65% dengan harga kalibrasi.

## 6. Generasi (praktik)

* temp 0,7: lancar, morfologi benar, generik, fakta campur-aduk (Jokowi…Ronaldo).
* temp 1,0: jatuh ke atraktor lintas-prompt ("Pada 9 Juni yang » MabcAX…").
* Aturan pakai: temp ≤0,7 + min-p 0,05 + sampel ≤300B + `ancestral=True`;
  `generate()` kini restore-konteks (order-independent, teruji).
* Belum ada: dropout-konteks (satu-satunya obat kolaps yang belum diuji),
  top-p, penalti repetisi.

## 7. Metodologi & disiplin (yang wajib dijaga)

Jendela uji TETAP tak-bertumpuk · replikasi = offset data (seed Tier A tak bermakna) ·
SE berpasangan + `per_win` tersimpan · kontrol model-ACAK tiap integrasi ·
CE per-BYTE · uji asap sebelum run panjang · edit kernel → golden+paritas dulu ·
run panjang sekuensial-detached (wrapper membunuh yatim; foreground >25 mnt berisiko).

## 8. Peta masalah → status

| # | masalah | status | berikut |
|---|---|---|---|
| 1 | Drift mixer / kurva-U | 🟢 fix V2 default (186M datar) | multi-offset G1-V2 final |
| 2 | Kapasitas tabel | 🟡 4-way terbukti (−0,197) tapi 0,73× | optimasi pindai / tb27 |
| 3 | Generasi kolaps | 🔴 terukur, obat parsial (min-p) | dropout-konteks |
| 4 | Tier B | 🔴 ditutup (G2 gagal) | backend dulu bila dibuka |
| 5 | Benjolan 18M | 🟡 transien-data, ambang | terima / kurva rapat |
| 6 | Biaya V2 ≤1M | 🟡 −0,008, 1 seed | multi-offset |
| 7 | Tech-debt kernel | 🟢 tuntas (freeze/ckpt/lr2/ctx/cks) | `wsel_mode=1` skala besar |
