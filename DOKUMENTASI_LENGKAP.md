# TMT2 v2 — Dokumentasi Lengkap

**Model bahasa byte-stream online, numpy murni (tanpa torch/scipy), CPU-only.**
Dokumen ini adalah rujukan tunggal dan konsolidasi dari `ARCHITECTURE.md`, `ARCHITECTURE_V2.md`,
`STATUS.md`, `LAPORAN_LENGKAP.md`, `README.md`, dan `FLOWCHART.md`. Semua angka di sini sudah
diukur; yang belum diukur ditandai eksplisit.

Terakhir diperbarui: **2026-10-09**. Total suite tes: **124 lolos**.
> 🟢 **Mixer v3** (fix kurva-U) tervalidasi 30M — rincian `HASIL_E7B_V3.md`,
> status terbaru `STATUS.md` §21. Rekomendasi skala: <1M legacy · ≥10M V3
> (`--mixer_mode 1 --sched 50000`). Default produksi masih legacy.
> 🔴 **Masalah terbuka saat ini** (kurva-U Tingkat A, G2 Tingkat B) dirangkum di
> **`MASALAH_SAAT_INI.md`** dan `STATUS.md` §20.

---

## Daftar isi

1. [Ringkasan satu halaman](#1-ringkasan-satu-halaman)
2. [Aturan main proyek (wajib)](#2-aturan-main-proyek-wajib)
3. [Arsitektur v2 — tiga tingkat](#3-arsitektur-v2--tiga-tingkat)
4. [Tingkat A — inti statistik biner](#4-tingkat-a--inti-statistik-biner)
5. [Tingkat B — neural paralel-chunk](#5-tingkat-b--neural-paralel-chunk)
6. [Tingkat C — Mixer-2 dan APM/SSE](#6-tingkat-c--mixer-2-dan-apm-sse)
7. [Hasil terukur (semua angka jujur)](#7-hasil-terukur-semua-angka-jujur)
8. [Gerbang pra-terdaftar G0–G5](#8-gerbang-pra-terdaftar-g0g5)
9. [Jebakan protokol — yang tidak boleh diulang](#9-jebakan-protokol--yang-tidak-boleh-diulang)
10. [Klaim yang DITARIK](#10-klaim-yang-ditarik)
11. [Cara menjalankan (perintah persis)](#11-cara-menjalankan-perintah-persis)
12. [Peta paket, skrip, dan dokumen](#12-peta-paket-skrip-dan-dokumen)
13. [Status & rencana berikutnya](#13-status--rencana-berikutnya)
14. [Masalah terbuka (kurva-U) & dua cacat kode](#14-masalah-terbuka-kurva-u--dua-cacat-kode)

---

## 1. Ringkasan satu halaman

TMT2 adalah model yang belajar **online**, satu byte demi satu byte, memprediksi byte ke-t
sebelum melihatnya (**protokol prekuensial**). Metrik tunggal: **BPB** (bit per byte).

Rancangan v2 memecah beban jadi tiga tingkat:

| tingkat | isi | berkas | kecepatan | BPB (WikiText) |
|---|---|---|---|---|
| **A** | inti statistik biner (tabel hash + mixer + APM) | `tmt2/stat_core.py` | **~59–100k B/s** | **1,78–1,83** |
| **B** | neural paralel-chunk (Gong/Saron) | `tmt2/gamelan/` | ~1–3 kB/s (latih, numpy) | tidak dipakai (G2 gagal @1 jt byte) |
| **C** | Mixer-2 + APM/SSE (bagian dari A) | di dalam `stat_core.py` | — | — |

**Temuan inti proyek ini, jujur:**

1. **Yang memberi kualitas adalah MIXER konteks, bukan rekurensi.** SSM/rekurensi menyumbang
   **≈0,000 bit** di atas mixer begitu mixer punya orde ≥3 (§7.2).
2. **Tingkat A (tabel biner + mixer + APM) mengalahkan mixer simbol lama** di kedua korpus
   dan **mengalahkan SEMUA kompresor umum** (lzma/bz2/zlib) pada korpus Indonesia
   **30 jt byte: 2,1865** (§7.3, §7.6).
3. **Tingkat B (neural) GAGAL gerbang G2** — diuji ulang pada skala penuh 1 jt byte:
   Δ = **−0,0005** (kontrol acak bersih). Dikeluarkan dari jalur produksi (§5, §8).
4. **Skala turun monoton** (G4 LULUS) hingga 1,1 MB; diperluas ke **30 jt byte Indonesia (2,1865)**.
5. **Shard multi-inti menaikkan kecepatan (4,6× @8 inti) tapi TIDAK menaikkan kualitas** (§7.5).
6. 🔴 **Masalah terbuka**: **kurva-U** — pada korpus bergerak, BPB-uji **naik** setelah tabel jenuh
   (terbukti **internal**, bz2 per blok datar). Lihat §14 dan `MASALAH_SAAT_INI.md`.
7. **Dua cacat kode nyata** ditemukan & diperbaiki (checksum match = bit indeks; indeks konteks
   tak masuk hash) → **−0,0616 bit**. Lihat §14.

---

## 2. Aturan main proyek (wajib)

Aturan ini lahir dari bug yang berulang. Semuanya berlaku mutlak.

### 2.1 Metrik & protokol
- **Metrik tunggal: BPB (bit/byte)**, bukan akurasi atau perplexity.
- **Protokol prekuensial**: prediksi byte ke-t sebelum melihatnya. Tidak ada kebocoran masa depan.
- **Jendela uji TETAP.** Semua baris dalam satu perbandingan harus dinilai pada jendela uji yang
  SAMA. Angka dari jendela uji berbeda **tidak boleh dibandingkan** (§9.5).
- **Angka ONLINE vs BEKU tidak boleh dibandingkan** — jendelanya berbeda.
- **≥3 seed** untuk setiap klaim kualitas. Baseline dijalankan pada protokol yang sama persis.

### 2.2 Data
- **Data acak WAJIB lewat `corpus.random_bytes(n, vocab, seed)`.** JANGAN `bytes(rng.integers(...))`
  — itu memakai buffer mentah `int64` → 8n byte dengan 7/8 nol. Bug ini pernah memunculkan dua
  temuan palsu. Dijaga `tests/test_corpus.py`.

### 2.3 Lingkungan teknis
- **Runtime**: `C:/Users/aamss_9izqos3/.workbuddy-ai/binaries/python/envs/default/Scripts/python.exe`
  (numpy 2.5.3, pytest 9.1.1, Python 3.13.12). Mesin uji **8 core**.
- **Tes**: `pytest -q` dari akar proyek. Status: **124 lolos**.
- **numba**: jangan `@`/`np.dot` 2-D di dalam `@njit` (butuh scipy) → tulis loop eksplisit.
  **Jangan tambah scipy.** numpy `@` di luar numba (BLAS) boleh.
  ⚠️ **Konstanta tingkat-modul DIBEKUKAN numba** — jangan diubah saat runtime; jadikan parameter
  fungsi (lihat §9.12).
- Kuirk numba: `int(np.uint64(x))` → **float64** (pakai `np.int64`); tabrakan NAMA variabel di
  kernel panjang → galat menunjuk baris yang salah.

### 2.4 Verifikasi gradien
- **Gradcheck**: pakai **turunan berarah** (satu arah acak per tensor), bukan per-entri
  (entri mendekati nol → rel palsu ~1).
- Untuk tes gradien vs beda-hingga: `adapter_rank=0` + `slow_every` besar.
- **Uji SATU langkah**, bukan lintasan latih panjang (online learning kacau: 1e-16 → 1e-5
  dalam 2900 langkah).

### 2.5 Benchmark
- Aliran **≥10k byte**, laporkan **rasio**, **multi-seed**, model baru setiap rep.
- Mesin dipakai bersama → B/s berayun ~1,5× antar-jam → laporkan **best-of-N**, jangan kutip
  satu angka sendirian.
- **Uji asap cepat SEBELUM latih panjang** (bug generasi ketangkap justru karena ini, §9.7).

---

## 3. Arsitektur v2 — tiga tingkat

```
                    byte masuk (8 keputusan biner)
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
   ┌────▼─────┐        ┌──────▼──────┐       ┌──────▼──────┐
   │ TINGKAT A│        │  TINGKAT B  │       │  (opsional) │
   │ tabel    │        │  neural     │       │  Bonang     │
   │ hash +   │        │  Gong/Saron │       │  (attn)     │
   │ konteks  │        │  paralel-   │       │  belum ada  │
   │ + match  │        │  chunk      │       │             │
   └────┬─────┘        └──────┬──────┘       └──────┬──────┘
        │                     │                     │
        │  stretch(p(bit))    │ stretch(p(bit|prefiks))
        │                     │  (lewat pohon biner)
        └──────────┬──────────┴─────────────────────┘
                   │
            ┌──────▼───────┐
            │  TINGKAT C   │  Mixer-2 berlapis (3 mixer → 1 campuran)
            │  + APM/SSE   │  2 tahap (256×33, 65536×33)
            └──────┬───────┘
                   │
              p(bit) → byte
```

**Dekomposisi biner adalah inti kecepatan.** Satu byte = 8 keputusan biner. Setiap konteks
memberi `p(bit)`, dan masukan mixer adalah `stretch(p) = ln(p/(1−p))`. Biaya per byte adalah
**O(bit × jumlah konteks)** ≈ 8 × 15, bukan **O(V × orde)** seperti mixer bertingkat-simbol.
Itulah sebabnya Tingkat A ~35× lebih cepat dari jalur simbol.

**Dualitas chunkwise (trik untuk Tingkat B).** Operasi rekuren O(1)/langkah berguna untuk
inferensi, tetapi latihannya terikat plafon RTRL. Bentuk *chunkwise* mengubah intra-chunk
menjadi matmul (BLAS) → memecah plafon itu, dengan syarat intra-chunk **benar-benar** matmul
(§9.6).

---

## 4. Tingkat A — inti statistik biner

Berkas: `tmt2/stat_core.py` (numba `@njit`). Ini **jalur produksi**.

### 4.1 Komponen

| komponen | rincian |
|---|---|
| **Tabel hash** | direct-mapped, checksum 8-bit, dipadatkan **satu int64 per slot**: `(chk<<32)|(n0<<16)|n1` |
| **Penghitung** | 16-bit, laju **adaptif** (pangkas `>>1` saat jenuh > `COUNT_CAP=1023`) |
| **Konteks** | pola offset (`build_contexts`): dense `0..6,8,12,16,24` + **konteks kata** + konteks jarang |
| **Model match** | konteks `minlen` byte → byte penerus pada kemunculan terakhir; `match_conf` = besar sinyal |
| **Mixer-2** | lapis 1: 3 mixer (pemilih `c0`, `prev_byte`, `(prev_byte,c0)`); lapis 2: campuran, dipilih `c0` |
| **APM/SSE** | 2 tahap (256×33 dan 65536×33), 33 bucket, interpolasi |

Konteks kata ditandai sentinel `coff[c,0] = -2`; `-1` = tak dipakai; `clen==0` = konteks kosong
(order-0). **Jangan satukan ketiganya** (§9.4).

### 4.2 Optimasi kecepatan (semua BIT-IDENTIK — BPB tidak berubah)

| optimasi | efek | alasan |
|---|---|---|
| hash konteks **diangkat keluar loop bit** | 50k → 75k B/s | konteks byte tak bergantung bit, tapi dulu dihitung 8× |
| tabel dipadatkan jadi **satu int64** | 75k → 91k | 3 akses memori acak → 1 (tabel ratusan MB) |
| simpan **`key`**, bukan slot+checksum | 91k → ~100k | separuh trafik memori |

### 4.3 Optimasi mixer
`stretch(squash(d)) == d` → keluaran lapis-1 cukup logitnya, **tanpa `exp`**. Semua mixer lapis-1
dilatih dengan galat **akhir** (gaya lpaq) → hanya **satu `exp` per bit**.

### 4.4 Ablasi Tingkat A (WikiText)

| komponen | efek pada BPB |
|---|---|
| tabel 2²² → 2²⁶ | **−0,16** |
| **konteks jarang (sparse)** | **+0,038 s.d. +0,057 (MERUGI — `stat_sparse=False`)** |
| APM | −0,034 |
| Mixer-2 | −0,012 |
| konteks kata | −0,012 |
| pemilih bobot 8 → 12 bit | −0,023 |
| model match | −0,008 |
| order-0 | −0,003 |

> **Catatan koreksi (2026-10-04)**: `bench_stat_core.py` dan `scale_curve.py` semula default
> **sparse ON**, padahal ablasi menunjukkan sparse MERUGI. Default keduanya sudah diperbaiki jadi
> **sparse OFF**. Diukur ulang pada jendela uji tetap : sparse OFF **2,3185** vs ON 2,3755
> (**−0,057 bit**). Semua angka produksi di dokumen ini sudah memakai sparse OFF.

### 4.5 API

```python
from tmt2.stat_core import StatCore
m = StatCore(orders=(1,2,3,4,5,6,8,12,16,24), table_bits=25, lr=0.002,
             use_apm=True, wsel_bits=12, sparse=False, word=True)
for b in data: m.step(b)                    # latih
ce = m.eval_stream(held_out)                # CE prekuensial (beku, konteks tetap jalan)
txt = m.generate("prompt", n=300, temp=0.7, seed=0)
```

---

## 5. Tingkat B — neural paralel-chunk

Berkas: `tmt2/gamelan/` — `gong.py`, `saron.py`, `block.py`, `train.py` (acuan numpy float64).

### 5.1 Komponen

- **Gong** — rekurensi linear dengan peluruhan bergantung-data:
  `U_t = λ_t ⊙ U_{t−1} + (1−λ_t) k_t v_tᵀ ; o_t = q_t U_t`.
  Dua bentuk: **rekuren** (O(1)/langkah, inferensi) dan **chunkwise** (intra-chunk matmul, latihan).
- **Saron** — konvolusi kausal (k=8) + SiLU, state dibawa antar-chunk.
- **Blok** — Saron + Gong + gerbang per head + SwiGLU + RMSNorm.

### 5.2 Tes paritas (semua lulus)

| tes | syarat | hasil |
|---|---|---|
| #1 Gong chunkwise == rekuren | < 1e-10 | **9,15e-14** (C = 1…256) |
| #2 Saron state antar-chunk == deret penuh | eksak | **6,66e-16** |
| #3 blok paralel == decode sekuensial | < 1e-9 | **lulus** |
| #4 gradien vs beda-hingga | rel < 1e-6 | **2,23e-07** |

### 5.3 Kecepatan (1 blok, d=32, H=4, T=2048, acuan numpy)
chunkwise C=64 = **14,8 µs/byte (67.628 B/s)** vs rekuren 57,2 µs/byte → **~4×**.

### 5.4 Hasil M5 dan putusan G2

Integrasi: keluaran Tingkat B → `stretch(p(bit|prefiks))` (pohon biner) → **satu masukan mixer**
Tingkat A. Skrip: `scripts/m5_integrate.py`.

| skala | A saja | A + B | Δ |
|---|---|---|---|
| 30k byte, 3 epoch | 3,0716 | 3,0632 | **+0,0084** |
| 100k byte, 5 epoch | 2,9754 | 2,9909 | **−0,0155** |
| **kontrol: B ACAK** | 3,4503 | 3,4513 | **−0,0011** |

**Diuji ULANG pada skala penuh (2026-10-05/06)** — skrip `scripts/tierb_integrate.py`,
**1.000.000 byte** Indonesia, tb=25, jendela uji tetap 50 kB:

| varian | BPB | Δ vs A |
|---|---|---|
| A saja | 3,2159 | — |
| A + B terlatih | 3,2164 | **−0,0005** |
| A + B acak (**kontrol**) | 3,2161 | −0,0001 → protokol **bersih** |
| A + B kocok (**kontrol**) | 3,2324 | −0,0165 |

**PUTUSAN: G2 BELUM LULUS** (ambang +0,05) — kini terkonfirmasi pada skala yang dokumen minta
(≥1 jt byte). Neural membawa **sinyal temporal** (kocok merugikan 0,017) tetapi **tidak menambah
informasi**. Sesuai §7 dokumen: Tingkat B **dikeluarkan dari jalur produksi**, modul riset.

**Tingkat B mandiri**: 250k/8k Indonesia = **2,6654** (Tingkat A korpus sama **2,3261**) → kalah
0,34 bit; CE masih turun di epoch terakhir. ★ **Generasi runtuh ke tarikan tetap** (prompt-independent
setelah ~40 byte) — **bukan** bug train/eval (fungsi latih vs generasi identik, selisih 1,7e−15).
★ **Sebaran τ** (`bl=0` → λ=0,5 semua head) diimplementasikan & diukur: 2,7300 vs 2,6654 →
lebih buruk, tidak memperbaiki keruntuhan; default = pemenang terukur (flag `tau_spread`).

---

## 6. Tingkat C — Mixer-2 dan APM/SSE

Tingkat C hidup **di dalam** `stat_core.py` (bukan berkas terpisah):

- **Lapis 1**: 3 set bobot, masing-masing dipilih konteks kecil berbeda (`c0`, `prev_byte`,
  `(prev_byte,c0)`).
- **Lapis 2**: campuran ketiga keluaran, dipilih `c0`.
- **Pembaruan online**: `w += η · (y − p) · x_i`; gradien sederhana & eksak.
- **APM/SSE**: 2 tahap, interpolasi 33 bucket, rata-rata akhir dengan prediksi mixer.

**Jebakan APM**: APM harus **DITERAPKAN** saat eval (hanya tabelnya tidak diperbarui).
Menggerbang `use_apm` pada `train` membuat APM tak berpengaruh sama sekali.

---

## 7. Hasil terukur (semua angka jujur)

### 7.1 Konfigurasi produksi
`table_bits=25` (WikiText/1M) · **`table_bits=26` untuk korpus besar (30 jt byte)**, `wsel_bits=12`,
`cks_bits=24`, APM on, match on, **word on, sparse OFF**, `match_fix=True`, `ctx_fix=True`.
→ **`table_bits=26` lebih baik dari 25 pada korpus besar**: pada 14 jt byte tb=26 CE **1,978
(masih turun)** vs tb=25 2,048 (sudah berbalik naik). Tabel 2× lebih besar menunda penimpaan
(lihat §14 kurva-U).

### 7.2 Ablasi SSM: rekurensi menyumbang NOL
WikiText, d=32, L=2, byte, CE state dingin (200k latih):

| varian | BPB |
|---|---|
| SSM saja | 2,9235 |
| **mixer saja** | **1,8434** |
| mixer + SSM | 1,8440 |

→ **SSM menyumbang ≈0 di atas mixer.** SSM hanya mengisi konteks orde rendah:
mixer orde-1 ± SSM = 3,0370 / 2,9707 (**+0,066**); orde-2 +0,009; orde-8 **0,000**.
**Jadi SSM ≈ model konteks orde-2,5**, redundan begitu mixer punya orde ≥3.

### 7.3 Tingkat A vs baseline (protokol G1: latih 200k / uji 8k)

| model | WikiText | Indonesia | kecepatan |
|---|---|---|---|
| **Tingkat A** | **1,7837** | **2,3261** | **~100k B/s** |
| TMT2 + mixer simbol | 1,8264 | 2,4809 | 2.666 B/s |
| n-gram JM-5 beku | 1,9254 | 2,5400 | — |
| **bz2 -9 kondisional** | 1,9450 | 2,3700 | — |
| lzma | 2,1480 | 2,4040 | — |
| BPE token 1-gram adil | 2,2129 | — | — |
| zlib | 2,5630 | — | — |

- WikiText: Tingkat A menang **+0,043** atas mixer simbol (ambang G1 = +0,05, **meleset 0,007**;
  tabel 2²⁶ memberi 1,7784 = +0,048).
- Indonesia: Tingkat A menang **+0,155** atas mixer simbol dan **+0,044 atas bz2 -9**.

### 7.4 Kurva skala G4 (LULUS)

WikiText, konfigurasi produksi (**sparse OFF**), **jendela uji tetap** (8.000 byte terakhir):

| ntrain | BPB (produksi) | BPB (sparse ON, lama) | Δ vs prev |
|---|---|---|---|
| 100k | **2,6098** | 2,6678 | — |
| 200k | **2,3185** | 2,3755 | −0,2913 |
| 400k | **2,1662** | 2,2046 | −0,1523 |
| 800k | **2,0973** | 2,1263 | −0,0689 |
| 1,10M | **2,0286** | 2,0630 | −0,0687 |

**Monoton turun, tanpa plateau.** Sparse OFF lebih baik di setiap skala (−0,03 s.d. −0,06 bit).
Throughput ~90k B/s. (10M/100M belum — korpus lokal maks 1,12 MB.)

### 7.5 Shard multi-inti (kecepatan naik, kualitas runtuh)

WikiText 200k/8k, `table_bits=22`, `wsel_bits=12`:

| N inti | split BPB | full BPB | full agregat B/s | speedup |
|---|---|---|---|---|
| 1 | 2,4437 | 2,4437 | 30.974 | 1,00× |
| 2 | 6,1432 | 5,9888 | 44.075 | 1,42× |
| 4 | 12,1593 | 11,5492 | 62.036 | 2,00× |
| 8 | 21,3100 | 19,3525 | **143.172** | **4,62×** |

**Shard = pengungkit KECEPATAN penilaian, BUKAN kualitas.** Penyebab: inti biner sangat jenuh
(16,5% bit di-clamp 1e-6) dan deterministik (dua model `lr` berbeda → BPB identik 2,5077/2,5076)
→ tak ada keragaman untuk di-ensembel. G3 (≥500k agregat) belum tercapai.

### 7.6 Latih + generasi korpus Indonesia 1M

Skrip `scripts/train_gen_id.py`. Latih **seluruh 957.490 byte**:

| metrik | nilai |
|---|---|
| BPB tahan (uji tetap 8.000 B) | **1,8046** |
| waktu latih | **12,6 s** |
| throughput | ~80.000 B/s |
| tabel | 2²⁵ ≈ 268 MB |

**Generasi** (temp 0,75): morfologi & sintaks lokal Indonesia baik; koherensi topik jarak-jauh lemah
(batas wajar n-gram byte orde ≤24).

> *"Indonesia adalah negara kepulauan yang diantaranya adalah bahasa. Remiliki sistem kompleks, dan
> dalam di tempat, khususnya sendiri sitosol disebut sebagai bentuk dari dan menghindari... Sementara,
> dan mengandung dari terbatas pada masa yang bersama kalangkan dengan memiliki seluler sel dia
> eukariotik."*

Laporan HTML: `results/gen_id_tierA_report.html`.

### 7.7 Angka dasar yang boleh dikutip (WikiText 200k, bit/byte)
bz2 -9 **2,2496 (target)** · lzma 2,4406 · ctx-mix-3 2,4095 · zlib 2,7570 · byte n-gram o2 2,724 ·
BPE token 1-gram adil 2,115. **Lisensi 46.507 B**: o0 4,561 · o1 3,650 · o2 2,849 · gzip -9 2,633.

### 7.8 Model 30 jt byte korpus Indonesia + baseline ADIL (2026-10-05/06)

Skrip `scripts/train_gen_big.py`. Latih **30.000.000 B** `data/indonesian_big.bin` (tb=26, cks=24,
APM on, sparse off, fix ON, orders 1..24): **505 s · ~59k B/s · tabel 100% terisi**.

| metode (jendela uji SAMA: 200 kB setelah prefix latih) | BPB | TMT2 − metode |
|---|---|---|
| **TMT2 Tingkat A** | **2,1865** | — |
| lzma -6 (beku) | 2,2680 | **−0,082** |
| bz2 -9 (beku) | 2,4572 | −0,271 |
| zlib -9 (beku) | 3,2888 | −1,102 |

→ **Menang atas SEMUA kompresor umum**, termasuk lzma (terkuat). Checkpoint
`checkpoints/id_big_30m.npz` (293 MB terkompresi, 67,1 jt slot) — round-trip load terbukti
(`scripts/gen_from_ckpt.py`). Laporan: `results/report_30m.html`; teks: `results/gen_big_id_30m.txt`.
Generasi temp 0,5: lancar & bermorfologi benar (contoh: *"…Sulawesi Tengah berjalan tentang pihak…"*).

---

## 8. Gerbang pra-terdaftar G0–G5

| gerbang | kriteria | status |
|---|---|---|
| **G0** alat ukur | profil produksi tercatat; bench satu perintah; baseline kondisional | **LULUS** |
| **G1** Tingkat A | ≥ +0,05 bit atas mixer simbol (WikiText 200k/8k, 3 seed) + tutup gap bz2 (Indonesia) + ≥100k B/s | **2 dari 3** (kecepatan LULUS · Indonesia LULUS · WikiText +0,043 meleset 0,007). Diperkuat: 30 jt byte Indonesia **2,1865**, menang atas lzma/bz2/zlib |
| **G2** nilai neural | Tingkat B ≥ +0,05 bit di atas A pada ≥1M byte, **tak mengecil** | **BELUM LULUS** — diuji ulang @1 jt byte: **Δ −0,0005** (kontrol acak bersih) |
| **G3** kecepatan | ≥500k B/s agregat + pembelajaran multi-shard | **sebagian** (143k B/s @8 inti, kualitas runtuh; Tingkat B ~1–3 kB/s) |
| **G4** skala | 1M→10M→100M turun monoton tanpa plateau | **LULUS** pada 100k→1,1M; diperluas ke 30 jt byte Indonesia (2,1865). ⚠️ **kurva-U** mulai muncul di 10–30 jt byte (lihat §14) |
| **G5** paritas | semua tes lulus | **124 lolos** |

**Aturan gerbang**: bila gagal → sesuai kolom tindakan (debug / keluarkan dari produksi /
laporkan jujur / selidiki). **Jangan klaim ulang.**

---

## 9. Jebakan protokol — yang tidak boleh diulang

Ini daftar kebocoran & bug yang sudah terdeteksi. Melanggarnya menghasilkan temuan palsu.

### 9.1 Tabel beku ≠ KONTEKS beku (kebocoran #1)
Saat eval beku, `hist` (riwayat byte) **WAJIB tetap berjalan**; hanya COUNTS yang dibekukan.
Kalau konteks ikut dibekukan → CE palsu 12–17 bit.

### 9.2 Probe CE-latih (kebocoran #2)
Probe info-state WAJIB punya **split latih/uji sendiri**. Versi pertama mengevaluasi di himpunan
LATIH → angka "terlalu bagus" (0,34 bit) = sinyal bug. Angka probe absolut pesimistis (himpunan
kecil); hanya perbandingan RELATIF yang sah.

### 9.3 Off-by-one jalur integrasi (kebocoran #3)
`nn_st[i]` dulu dibuat dari state yang **sudah melihat** byte i, padahal dipakai memprediksi byte i;
pohon biner memakai prefiks yang sudah dilihat. Versi pertama M5: +0,143 terlatih, **+0,260 ACAK**.
**PELAJARAN: model ACAK adalah kontrol WAJIB untuk setiap jalur integrasi.** Kalau model acak
"menang", protokolnya yang bocor.

### 9.4 Jendela uji berpindah (kebocoran #4)
`raw[n:n+ntest]` → tiap baris kurva dinilai di jendela BERBEDA → goyangan palsu ±0,09 bit,
tampak "tidak monoton". **Pakai SATU jendela uji tetap untuk semua perbandingan.**

### 9.5 Angka ONLINE vs BEKU tak boleh dibandingkan
Jendelanya berbeda (online: riwayat menyerap jendela uji; beku: tidak). Bandingkan hanya dalam
mode yang sama.

### 9.6 Paritas saja tidak cukup — kecepatan wajib diukur
`gong_chunkwise` versi pertama memakai loop O(C²) → **12× LEBIH LAMBAT** dari rekuren
(738 µs/byte). Dualitas chunkwise hanya berguna bila intra-chunk benar-benar matmul.

### 9.7 Bug satuan & bug generasi (dua terbaru)
- **Satuan CE**: rata-rata CE harus per **BYTE** (×8), bukan per bit. Laporan "BPB 0,315" = tepat
  `2,522/8`. Tampak seperti kebocoran, ternyata lupa faktor 8.
- **`_sample_byte`** dulu `np.prod(byte_probs(b))` — mengalikan **p(bit=1) untuk SEMUA bit**
  seolah tiap bit bernilai 1 → byte biner >127 menang → generasi keluar byte sampah.
  **Perbaikan**: kalikan menurut **nilai bit byte itu** (`p` bila bit 1, `1−p` bila bit 0).
  Rumus benar sudah ada di tes lain tapi tidak diikuti kode produksi.
  **PELAJARAN**: jalankan **uji asap cepat sebelum latih panjang**; fungsi tanpa tes = celah.
  Dijaga 2 tes regresi baru.

### 9.8 Jebakan peluruhan numerik
Peluruhan relatif Gong WAJIB di ruang log (`exp(cumlog_i − cumlog_j)`), bukan membagi produk
kumulatif — pembagian underflow saat ada λ kecil (kasus NORMAL).

### 9.9 Windows multiprocessing = `spawn`
`Pool(initializer=...)` hanya bekerja bila skrip punya guard `if __name__ == "__main__"` dan
fungsi worker di level modul. Tes cepat via `python -c` gagal
(`Can't get attribute ... on __main__`).

### 9.10 Default skrip ≠ konfigurasi produksi
`bench_stat_core.py` dan `scale_curve.py` semula default **sparse ON**, padahal ablasi sudah
menunjukkan sparse MERUGI. Akibatnya angka yang dilaporkan memakai knob yang diketahui lebih buruk.
**PELAJARAN**: setelah sebuah ablasi selesai, **samakan default skrip dengan pemenangnya**; jangan
biarkan knob default menyimpang dari konfigurasi produksi. Diperbaiki 2026-10-04 (sparse OFF).

### 9.11 Jangan biarkan angka lama hidup berdampingan dengan protokol baru
Angka "Tingkat A WikiText 1,7837" diukur pada jendela uji **berbeda** dari kurva G4. Keduanya sah
*dalam protokolnya masing-masing*, tetapi tak boleh dibandingkan langsung (§9.4). Saat protokol
berubah, **ukur ulang** dan tandai angka lama sebagai milik protokol lama.

### 9.12 Konstanta tingkat-modul DIBEKUKAN numba (jebakan halus)
Mengubah konstanta tingkat-modul (mis. `COUNT_CAP`) saat runtime **tidak berefek** di dalam `@njit`
— nilainya dibekukan saat kompilasi. Terbukti: `COUNT_CAP` 1023 vs 4095 memberi BPB **identik**
(2,483178) → awalnya tampak "tidak berpengaruh", padahal knobnya **tidak pernah dibaca**.
**Perbaikan**: jadikan **parameter fungsi** (`count_cap`, `cks_bits`, …). Setelah itu efeknya
diukur ulang (tetap ≈0, tapi kini sah).

### 9.13 CE kumulatif menyesatkan pada latihan panjang
Rata-rata CE **dari byte ke-0** hampir tak bergerak saat latihan panjang; lonjakan segmen terakhir
(data sukar) tampak seperti model **mundur**. Ini memicu "kok makin jelek?" yang palsu.
**Perbaikan**: laporkan **CE-seg berjendela** (reset tiap laporan) + **CE-cum** + **BPB-uji berkala**.

### 9.14 `count_cap` besar menghambat adaptasi pada data non-stasioner (dugaan, belum diuji)
Penghitung dengan `count_cap=1023` bereaksi lambat; pada korpus bergerak ini bisa merugikan.
Uji **turun** (60–255) **belum** dilakukan — jangan klaim sampai diukur.

---

## 10. Klaim yang DITARIK

Jangan diulang — semuanya artefak protokol/bug, bukan temuan nyata:

| klaim ditarik | sebab |
|---|---|
| "38.540 B/s / 20k tercapai" | protokol parsial (tak pernah ukur satu byte penuh) |
| "mixer beku 1,138 bit" / "0,83 bit" | kontaminasi urutan eval (jujur ≈2,1–2,3) |
| "alias periode-8" | artefak bug `bytes(numpy)` |
| "latihan tumpukan merusak representasi" | artefak bug `bytes(numpy)` |
| "kontribusi SSM negatif, penalti 0,04→0,09" | lantai Markov salah, jendela uji beda, 1 seed |
| "kueri lewat `h` gagal" | masalahnya init `Wq`, bukan sumber kueri |
| "§14 angka beku 1,138/1,182" | audit kebocoran §9.1 |
| "hipotesis long-range retrieval" | hanya −0,020 bit di atas mixer (gugur) |
| "hipotesis BWT bz2" | H0(MTF(BWT)) = 2,58, lebih buruk dari mixer (gugur) |
| "**APM merugikan / matikan APM**" | **SALAH BACA ARAH.** Data: `tanpa_apm` − `penuh` **selalu positif** → APM **MENOLONG** (+0,0275 @5 jt → **+0,058 @30 jt**). Manfaat APM **tumbuh** seiring latihan. Klaim "matikan APM" **ditarik**. |

---

## 11. Cara menjalankan (perintah persis)

```bash
PY=C:/Users/aamss_9izqos3/.workbuddy-ai/binaries/python/envs/default/Scripts/python.exe

# tes lengkap (124 lolos)
$PY -m pytest -q

# --- Tingkat A: bench G1 ---
$PY scripts/bench_stat_core.py --corpora wikitext,indonesian \
     --ntrain 200000 --ntest 8000 --table_bits 25 --wsel_bits 12

# --- G4: kurva skala (jendela uji TETAP) ---
$PY scripts/scale_curve.py --corpus wikitext \
     --sizes 100000,200000,400000,800000,1100000 --table_bits 25 --wsel_bits 12

# --- shard multi-inti ---
$PY scripts/shard_stat_core.py --nshard 1,2,4,8 --ntrain 200000 --mode full

# --- latih + generasi Indonesia 1M (jalur produksi) ---
$PY scripts/train_gen_id.py --gen 700 --temp 0.75 --table_bits 25 --wsel_bits 12

# --- korpus besar: ekstraksi + latih 30 jt byte + generasi ---
$PY scripts/extract_parquet.py                       # parquet -> data/indonesian_big.bin
$PY scripts/train_gen_big.py --ntrain 30000000 --ntest 200000 --table_bits 26
$PY scripts/gen_from_ckpt.py --ckpt checkpoints/id_big_30m.npz   # muat ulang + generasi

# --- investigasi kurva-U ---
$PY scripts/diag_blocks.py --ntrain 40000000 --block 1000000      # CE per blok vs bz2
$PY scripts/summarize_blocks.py results/diag_blocks.jsonl
$PY scripts/diag_normalized.py --ntrain 30000000                  # urutan asli vs acak
$PY scripts/diag_match.py --ntrain 1000000                        # akurasi match + |w_match|

# --- baseline adil & papan skor ---
$PY scripts/gate1_sweep.py            # gate 1 penuh, 3 seed × 2 korpus, paralel
$PY scripts/fair_online_compare.py    # papan skor held-out
$PY scripts/strong_baseline.py        # bz2/lzma/ctx-mix

# --- Tingkat B: tes paritas, latih, integrasi ---
$PY -m pytest tests/test_gamelan.py -q
$PY scripts/tierb_train.py --ntrain 250000 --epochs 5          # B mandiri + generasi
$PY scripts/tierb_integrate.py --ntrain 1000000 --epochs 3     # A+B (putusan G2, 2 kontrol)
$PY scripts/m5_integrate.py                                    # integrasi lama (30k/100k)

# --- laporan ---
$PY scripts/make_30m_report.py · scripts/make_tierb_report.py · scripts/make_big_report.py
```

**Konvensi penting saat menjalankan**: `--ntest 8000` dengan jendela uji = ekor korpus.
Jangan pakai `raw[n:n+ntest]` (§9.4).

---

## 12. Peta paket, skrip, dan dokumen

### 12.1 Paket `tmt2/`

| berkas | isi |
|---|---|
| `autograd.py` | mini-autograd float64 (definisi kebenaran) |
| `core.py` | SSM + RTRL diagonal + kepala |
| `learner.py` | Stream/TrainOpts, jalur autograd |
| `fastpath.py` / `fast_learner.py` | numba, jalur cepat + kernel fused |
| `model.py` | `Config` + `TMT2` |
| `stat_core.py` | **Tingkat A + C** (inti statistik biner, numba) |
| `gamelan/gong.py`, `saron.py`, `block.py`, `train.py` | **Tingkat B** |
| `tokenizer.py` | pembungkus BPE |
| `corpus.py` | pemuat korpus (`license`/`periodic`/`wikitext`/`tinystories`/`synthetic`/berkas) |

**Preset**: `Config.best()` = konfigurasi **simbol lama** (mixer orde 1..12 + `mixer_match`).
v2 dipakai lewat `Config.v2()` (field `stat_*`, `mix2_*`, `apm_stages`, `nn_*`).

### 12.2 Skrip penting (kelompok)

- **Kualitas**: `probe_state_info`, `probe_epochs`, `probe_readout_capacity`, `ngram_ceiling`,
  `mixer_variants`, `kn_ceiling`, `ablation_mixer`
- **Skor adil**: `gate1_sweep`, `fair_online_compare`, `fair_bpe_split`, `strong_baseline`,
  `fair_baseline`, `fair_compare_big`
- **Tingkat A**: `bench_stat_core`, `scale_curve`, `shard_stat_core`, `train_gen_id`
- **Korpus besar**: `extract_parquet`, `train_gen_big`, `gen_from_ckpt`
- **Investigasi kurva-U**: `diag_blocks`, `summarize_blocks`, `diag_normalized`, `diag_match`,
  `diag_component`, `diag_overwrite`, `diag_fix`
- **Tingkat B**: `tierb_train`, `tierb_integrate`, `m5_integrate`
- **Integritas**: `audit_leak`, `bptt_control`, `bptt_gradcheck`
- **Generasi**: `generate_id`, `gen_quality`, `gen_quality_v2`, `make_gen_report`
- **Laporan**: `make_30m_report`, `make_tierb_report`, `make_big_report`

### 12.3 Dokumen

| dokumen | peran |
|---|---|
| **`DOKUMENTASI_LENGKAP.md`** (ini) | rujukan tunggal konsolidasi |
| **`MASALAH_SAAT_INI.md`** | **masalah terbuka saat ini (kurva-U, G2), terurut prioritas** |
| `STATUS.md` | catatan kronologis paling rinci (§17 kualitas, §18 gate, §19 v2/M6, **§20 terbaru**) |
| `ARCHITECTURE.md` + `ARCHITECTURE_V2.md` | rancangan lama & v2 (M0–M6) + hasil + **§11 kurva-U** |
| `LAPORAN_LENGKAP.md` | laporan utama riset (§18 kualitas, §19 produksi, **§20 terbaru**) |
| `FLOWCHART.md` | diagram alur |
| `README.md` | pengantar & setup |
| `KAGGLE.md`, `FUSED_KERNEL.md` | catatan lingkungan & kernel |

---

## 13. Status & rencana berikutnya

### Status saat ini
- **Posisi**: Tahap 0 (sandbox). **Jalur produksi**: **Tingkat A saja** (tb=26 untuk korpus besar).
- **Yang terbuka**: 🟡 validasi akhir V3 (multi-seed + 190MB) & **G2** (neural, sudah diputus: gagal).
- **Tes**: **124 lolos**.

### Langkah berikutnya (berurut, jujur)
1. **Validasi akhir V3** (`HASIL_E7B_V3.md`, `STATUS.md` §21) — multi-seed G1-V3 +
   190 MB penuh, lalu balik default. Kurva-U katastrofik SUDAH hilang (V3 datar 30M).
2. **Kapasitas tabel** (pengungkit kedua, bukan penyebab) + **`count_cap` turun** (60–255).
3. **Latih 190 MB penuh** bila (1)/(2) menolong + generasi + laporan akhir.
4. **Tutup 0,007 bit G1 WikiText** (tabel 2²⁶ → 1,7784, +0,048).
5. **G3**: agregat ≥500k B/s (lebih banyak inti, atau shard khusus penilaian-beku).
6. **Keputusan neural**: G2 gagal pada skala penuh → **pakai Tingkat A saja**; Tingkat B hanya
   dihidupkan kembali bila ada bukti baru (Bonang/attn, d/H lebih besar, backend GPU).
7. **Pengerasan produksi**: throughput, stabilitas numerik, determinisme lintas-mesin, serving.

### Prinsip yang tidak boleh dilanggar saat melanjutkan
- Ukur dulu, klaim kemudian (§2). Setiap gate punya kriteria pra-terdaftar (§8).
- Jendela uji tetap; **model acak sebagai kontrol**; uji asap sebelum latih panjang (§9).
- Semua angka negatif tetap ditulis (§10).

---

## 14. Masalah terbuka (kurva-U) & dua cacat kode (2026-10-05/06)

> 🟢 **Pembaruan 2026-10-09: kurva-U katastrofik SUDAH ada fix tervalidasi (Mixer v3).**
> Penyebab = drift mixer (galat global + clamp-ghost), kanal utama kolom jarang —
> BUKAN tabel (E7b). V3 datar 30M (tb26 2,0317, max|w1| ~1). Rincian: `HASIL_E7B_V3.md`,
> `STATUS.md` §21. Bagian di bawah ini riwayat diagnosisnya (tetap sah untuk legacy).

### 14.1 Kurva-U Tingkat A (legacy) — 🟡 FIX TERSEDIA (V3, opt-in)
Pada korpus Indonesia 190 MB, **BPB-uji turun lalu NAIK**: tb=25 **2,1191 (3 jt) → 2,1028 (9 jt)
→ 2,2744 (30 jt)**; semua `table_bits` konvergen **~2,28** (tabel lebih besar hanya menunda).

**Bukti INTERNAL** (`diag_blocks.py`, satu model, blok 1 MB vs bz2 -9 blok itu):

| blok | CE kita | bz2 blok | rasio | isi tabel |
|---|---|---|---|---|
| 0 | 2,2034 | 2,5545 | 0,863 | 63,5% |
| 3–5 | **2,02–2,04** (min) | ~2,56 | 0,79 | 97–99% |
| 30 | 2,0797 | 2,5657 | 0,811 | 100% |
| 67 | 2,3477 | 2,5792 | 0,910 | 100% |

Kuartal 1 vs terakhir: **CE +0,2771 · bz2 +0,0026 (datar) · rasio 0,800 → 0,907**.
→ **kesulitan data stabil; degradasi murni internal** (tabel jenuh → penimpaan).

**Dibuang**: ukuran tabel (penyebab tunggal) · `count_cap` (nol) · `cks_bits` (−0,0003) ·
`apm_lr` (−0,007) · korelasi slot↔checksum (≈0) · kesulitan data (bz2 datar) · urutan/drift
dokumen (kurva-U di urutan ASLI **dan** ACAK).
**Tersisa**: direct-mapped + penimpaan buta · miskalibrasi `count_cap` · rasio penulisan 4,5×/slot.
**Rinci**: `MASALAH_SAAT_INI.md` · `ARCHITECTURE_V2.md` §11.

### 14.2 Dua cacat kode NYATA (diperbaiki)
| # | cacat | bukti | perbaikan |
|---|---|---|---|
| 1 | **checksum match = bit indeks** → verifikasi match selalu benar | **100% dari 299.994 sampel** | `mck = (mkey >> 40) & mck_mask` |
| 2 | **indeks konteks `c` tak masuk hash** → konteks berbeda bertabrakan | kapasitas efektif turun | `key = _mix64(hctx[c] ^ c0*GOLDEN ^ (c+1)*CCTX)` |

**Efek gabungan −0,0616 bit** (tb=22, 5 jt byte: 2,6371 → 2,5755). Akurasi match 0,541 → 0,587.
Dijaga flag `match_fix`/`ctx_fix`. + tes regresi.

### 14.3 Model 30 jt byte + baseline adil
Lihat §7.8. **2,1865**, menang atas lzma (2,2680), bz2 (2,4572), zlib (3,2888). Checkpoint tersimpan
& round-trip terbukti.

---

*Dokumentasi ini menggantikan ringkasan yang tersebar. Untuk detail kronologis dan angka mentah,
lihat `STATUS.md` (paling rinci) dan `results/*.jsonl` (hasil mentah setiap eksperimen).*
