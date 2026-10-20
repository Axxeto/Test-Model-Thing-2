# DIAGNOSA KECEPATAN — "E1 graft & Tier B integration sangat lambat"

Tanggal: 2026-10-06. Pertanyaan user: *"Sepertinya E1 graft exp dan tier b integration sangat
lambat, apakah mungkin 1b/s atau kurang atau bahkan dibawah 10k b/s? Karena sangat lama, padahal
seharusnya bisa lebih dari itu atau bahkan 100k b/s."*

Jawaban singkat: **kecurigaan Anda BENAR, tetapi penyebabnya bukan kernel.** Kernel `stat_step`
sudah terkompilasi numba dan memang mencapai ~90-140k B/s. Yang lambat adalah **cara skrip
memanggilnya** — byte demi byte dari Python. Untuk Tier B, kelambatan itu **memang desainnya**
(loop Python per blok) dan bukan cacat skrip.

---

## 1. Angka TERUKUR (bukan perkiraan)

Diukur langsung pada mesin ini, korpus `data/indonesian_big.bin`, tb=25, 10 konteks:

| lapisan | laju | catatan |
|---|---|---|
| `StatCore.step` per byte (loop Python) | **~86-94 kB/s** | kernel numba jalan, tapi dipanggil dari Python tiap byte |
| `StatCore.eval_stream` (loop Python) | **~95-109 kB/s** | idem, mode beku |
| `StatCore.train_stream` (loop numba) | **~139 kB/s** | ✅ jalur baru |
| `StatCore.eval_stream_fast` (loop numba) | **~163 kB/s** | ✅ jalur baru |
| **Pola probe E2 lama** (per-konteks × per-byte Python) | **6,6 kB/s** | ❌ inilah biang lambat |
| `StatCore.instrument_stream` (loop numba) | **~114-172 kB/s** | ✅ jalur baru (naik ~17-26×) |

**Kesimpulan angka:** kecurigaan "<10k b/s" **tepat untuk E2** (6,6 kB/s). E1 tidak sampai serendah
itu, tapi tetap tertahan di ~90 kB/s karena loop Python — bukan di target ">100k B/s".

---

## 2. Akar masalah (satu, bukan tiga)

### E1 & E2 (dan hampir semua skrip latih lain)
Semua skrip ini menulis:

```python
for i, b in enumerate(tr):
    m.step(int(b), train=True)      # ← pemanggilan Python 30.000.000×
```

Setiap `m.step` menyeberangi batas Python↔numba. Numba cepat, **tapi penyeberangan batas itu tidak
gratis**: ~11 µs/byte. Itu yang mengunci ke ~90 kB/s. Yang lebih parah, probe E2 lama menambah loop
Python per-konteks (10 konteks) **di dalam** loop per-byte → 6,6 kB/s.

### Tier B (integrasi neural)
Ini lambat karena **arsitekturnya**: `gamelan/` menjalankan langkah blok di Python penuh
(numpy murni, tanpa numba untuk loop blok) — sudah tercatat sebagai **N-01** di audit. Laju
terukurnya ~1-3 kB/s. **Ini bukan bug skrip**; 1 juta byte memang butuh ~10-15 menit. Kalau ingin
Tier B cepat, harus (a) tulis ulang loop blok dalam numba, atau (b) pindah ke backend BLAS/GPU —
keduanya pekerjaan riset, bukan perbaikan cepat.

---

## 3. Perbaikan yang SUDAH dipasang

Ditambah 3 kernel terkompilasi di `tmt2/stat_core.py` (loop byte pindah ke dalam numba):

- `stat_stream(...)` → `StatCore.train_stream(data)` & `StatCore.eval_stream_fast(data)`.
- `stat_probe_hitrate(...)` + `stat_instrument(...)` → `StatCore.instrument_stream(...)`
  (seluruh E2 kini di numba).

Skrip `scripts/e1_graft.py` & `scripts/e2_instrument.py` sudah dialihkan ke jalur ini.

**Jaminan paritas:** jalur cepat diuji **bit-identik** dengan loop Python acuan
(`max|Δ| = 0,000e+00` pada latih & uji). Ada dua penjaga tes baru:
`test_fast_stream_bit_identical_to_byte_loop` dan
`test_instrument_stream_does_not_perturb_training`. **97 tes lolos.**

### Dua jebakan yang ditemukan saat memasang (dicatat agar tidak terulang)
1. **Kuirk numba:** `stat_step` punya 37 argumen; menulis `1 if train else 0` inline di call-site
   memicu `UnsupportedBytecodeError: CALL_FUNCTION_EX with **kwargs not supported`. Harus
   pra-hitung flag di variabel sebelum loop.
2. **Bocor state probe E2 (lama):** probe menulis byte ke ring buffer `hist`; memulihkan hanya
   `pos/hlen/prev` TIDAK cukup — isi `hist` harus dicadangkan & dikembalikan, kalau tidak hasil
   latih bergeser. Sudah diperbaiki + dijaga tes.

---

## 4. Biaya penuh E1/E2 pada ukuran nyata (tb=25, 30 jt byte)

| tahap | perkiraan waktu |
|---|---|
| E1 latih 30 jt byte | **~216 s** |
| E1 eval (8 cangkok × 20 jendela × 200 kB = 32 MB beku) | **~196 s** |
| **E1 total** | **~7 menit** (sebelumnya: ~50+ menit karena loop Python + eval lambat) |
| E2 latih 30 jt + 30 probe (interval 1 jt, probe 20 kB) | **~4-5 menit** (sebelumnya: >2 jam) |

Sebelumnya E2 praktis **tidak selesai** dalam waktu wajar (6,6 kB/s → 30 jt byte ≈ 75 menit hanya
untuk latih, ditambah probe). Itulah kenapa Anda melihatnya "sangat lama".
