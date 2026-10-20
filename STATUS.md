# STATUS TMT2 — masalah, kesimpulan, analisis

> **📖 Ringkasan konsolidasi ada di [`DOKUMENTASI_LENGKAP.md`](DOKUMENTASI_LENGKAP.md).**
> Dokumen ini adalah catatan kronologis paling rinci.

Terakhir diperbarui: 2026-10-09 (revisi 11 — Mixer v3: akar drift + fix tervalidasi
30M, kanal sparse, gerbang §7 substansi-lulus; default produksi BELUM dibalik).
§17 kualitas · §18 jalan ke produksi · §19 v2 (A/B/C) · §20 kurva-U ·
**§21 Mixer v3 + gerbang (terbaru)**. Rincian angka: `HASIL_E7B_V3.md`.
Semua angka di bawah **terukur**, bukan perkiraan.
Status tes: **124 lolos** (`pytest -q`).

> 🔴 **REVISI 10 — MASALAH SAAT INI (baca dulu).** Dua hal yang **belum selesai**:
> **(1) Kurva-U Tingkat A.** Pada korpus Indonesia 190 MB, BPB-uji turun lalu **naik lagi**
> (2,10 → 2,27) seiring latihan. Terbukti **degradasi INTERNAL**, bukan data: bz2 per blok
> **datar** ~2,56, rasio CE/bz2 naik 0,80 → 0,91. Penyebab dominan = **tabel jenuh (penimpaan)**;
> `table_bits=26` menundanya. **(2) Tingkat B tidak menambah apa pun** — G2 diuji pada skala
> penuh 1 jt byte: Δ = **−0,0005** (kontrol acak bersih). Tingkat B = modul riset.
> Dua **cacat kode nyata** ditemukan & diperbaiki (checksum match = bit indeks; indeks konteks
> tak masuk hash) → −0,0616 bit. Model 30 jt byte: **BPB 2,1865**, menang atas lzma/bz2/zlib.

> **REVISI 9 — TMT2 v2 (§19).** Tingkat A (inti statistik biner) jadi jalur produksi: WikiText
> 1,78–1,83 · Indonesia 2,33 (200k) / **1,80 (1M)** @ ~100k B/s, mengalahkan mixer simbol &
> bz2 -9 (Indonesia). Tingkat B (neural) **GAGAL G2**. G4 skala **LULUS**; shard = kecepatan, bukan
> kualitas. Ditemukan & diperbaiki bug generasi `_sample_byte` (+2 tes regresi).

> **REVISI 8 — KUALITAS (§17) + JALAN KE PRODUKSI (§18).** Arah "ekspresivitas SSM + jalur baca
> non-memori" **selesai diukur**: state SSM ≈ **n-gram orde-2,5**; SSM **redundan** begitu mixer
> punya orde ≥3 (kontribusi marginal ≈ 0). Jalur baca **bukan** hambatan, begitu pula gradien,
> rezim online, dan smoothing n-gram. Kualitas TMT2 = kualitas **mixer**. `Config.best()` kini
> mixer orde 1..12 + `mixer_match` (−0,016 bit, 3/3 seed). **§18**: 5 gate menuju produksi —
> **Gate 1 BELUM LULUS**: menang di SEMUA baseline pada WikiText (termasuk bz2 -9 kondisional
> 1,945 & BPE token 2,213), tetapi **KALAH dari bz2 -9 (2,370) & lzma -9 (2,404) pada korpus
> Indonesia** (TMT2 2,481); **Gate 2 (nilai SSM) juga masih MEMBLOKIR**.
> ⚠️ **Koreksi**: angka probe §17.1 versi pertama adalah CE LATIH — sudah diganti CE uji.

> ⚠️ **REVISI 7 — BACA DULU.** Kernel TERPADU (`fused_fwd`/`fused_bwd`/`fused_promote_mem`)
> **SELESAI dan bit-identik** dengan jalur lama (§Kecepatan). Klaim lama **"25,95 us/byte =
> 38.540 B/s"** **DITARIK**: prototipe `proto_fused_est.py` tidak pernah mengukur sebyte penuh;
> setelah diperbaiki = 39,98 us/byte. **End-to-end nyata = 8.054 B/s @d=32** (2,44x atas jalur
> lama). **Target 20k B/s TIDAK tercapai** dan bukan soal tuning (plafon ≈10–12k B/s @d=32;
> mundur didominasi jejak RTRL `3·d²`). Fokus dipindah ke **kualitas (BPB)**.

> ⚠️ **REVISI 5 — AUDIT KEBOCORAN §16**: angka mixer beku **1,138 bit** (§14) = **artefak
> kontaminasi**: jendela uji bocor ke tabel `_mix_ng` karena urutan eval lama (online sebelum beku).
> **Klaim "0,83 bit" dan "1,138 = angka terbaik" DITARIK.** Angka beku jujur ≈ **2,1–2,3 bit**;
> efek orde jujur ≈ **0,28 bit** (dari kolom online). Kolom **online** tidak tercemar.
> Rincian: `LAPORAN_LENGKAP.md` **§16**.

> **Lihat juga `LAPORAN_LENGKAP.md`** — dokumen paper-style yang lebih lengkap (arsitektur,
> hierarki masalah H1–H4/M/S/U, semua perbaikan, jebakan metodologi). Berkas ini ringkasannya.
> **§12 LAPORAN = Pilar 2 (mixer konteks)**: kontribusi marginal SSM **negatif di 500k,
> terkonfirmasi 3 seed** (−0,0150 online / −0,0144 warm / −0,0143 cold, 9/9 sel; kolom online sah).
> **§13 = kontrol truncated BPTT**: hambatan **BUKAN algoritma belajar** (RTRL 0,857 < BPTT 0,983
> pada LR terbaik masing-masing) → **arsitektur/pemodelan**.
> **§14 = peningkatan mixer** ⚠️ kolom beku **TERKONTAMINASI** (lihat §16); kolom online jujur.
> **§16 = AUDIT KEBOCORAN**: 1,138 DITARIK; angka beku jujur ≈ 2,1–2,3.

**Pembaruan 2026-10-03 (M1–M4 SELESAI)**: default `Config` sudah dibalik ke konfigurasi terbaik —
`head="linear"`, `mem_query_x=True`, `mem_wq_match_k=True`; ditambah `Config.best()` (kini
mengaktifkan mixer orde 1..8). `README.md` diperbarui. Bug `scripts/bigtest.py`
(`--head` default menimpa Config) diperbaiki.
**Peringatan**: angka lama di bawah (mis. 4,970 kosinus) kini butuh flag eksplisit untuk direproduksi.

**Pembaruan 2026-10-03 (lanjutan) — ⚠️ DIKOREKSI**: klaim "§14 mixer orde 1..8 → BPB beku
1,138 bit" **DITARIK** (terkontaminasi, §16). Angka itu berasal dari eval beku yang dijalankan
**setelah** eval online menyerap jendela uji ke tabel n-gram. Angka beku jujur untuk mixer-murni
≈ **2,1–2,3 bit**; efek orde jujur ≈ **0,28 bit** (kolom online). Urutan eval diperbaiki
(default `eval_order=frozen_first`) + 2 tes penjaga. **§13 kontrol BPTT tetap berlaku.**

---

## 0. Verdict singkat

> ⚠️ **Verdict di bawah berlaku untuk JALUR LAMA (SSM simbolik + RTRL).** Sejak **TMT2 v2
> Tingkat A** (§19–§20), kesimpulannya **berubah**: Tingkat A **menang atas semua kompresor umum**
> pada korpus Indonesia 30 jt byte (**2,1865** vs lzma 2,2680 · bz2 2,4572 · zlib 3,2888).
> Masalah terbuka sekarang = **kurva-U** (§20.10, `MASALAH_SAAT_INI.md`), bukan "kalah dari n-gram".

**TMT2 (jalur lama) layak untuk eksperimen mekanisme, belum layak untuk menghasilkan model bahasa yang berguna.**

Alasannya satu kalimat: model ini **kalah dari baseline n-gram sederhana** pada teks nyata, dan
kontribusi mekanisme SSM-nya sendiri hanya **~0,01 bit**. Segala upaya memperbaiki memori (termasuk
memperbaikinya secara penuh di oracle) **tidak menggerakkan BPB teks alami** — jadi hambatannya
**bukan di memori**.

### Satuan dan protokol (dinyatakan eksplisit)

- **Semua angka kualitas = BPB = bit per byte** (bukan nat). `CE/np.log(2)`.
- **Protokol**: prekuensial (prediksi byte ke-t sebelum melihatnya), satu lintasan, tanpa
  shuffle. Angka "kumulatif" = rata-rata sepanjang aliran; "1/3 akhir" = rata-rata sepertiga
  terakhir (menghindari bias pemanasan).
- **Baseline dihitung pada byte yang sama**, n-gram adaptif dengan back-off. Lisensi
  (46.507 B, GPL-3 + Apache-2.0):

  | baseline | bit/byte |
  |---|---|
  | order-0 (frekuensi byte) | 4,561 |
  | order-1 | 3,650 |
  | order-2 | 2,849 |
  | **gzip -9** | **2,633** |

  TMT2 pada korpus yang sama: 4,970 → 4,924 (`head=linear`) → **3,794** (`deep_grad`),
  1/3 akhir **3,436**. Jadi **3,794 masih di atas order-2 (2,849) dan jauh di atas gzip**.
  Yang menang hanya 1/3 akhir (3,436 < 3,650 order-1) — dan itu pun belum menyentuh order-2.
- **Seed**: angka besar di bawah umumnya **satu seed** kecuali disebut lain. Hasil dengan
  klaim perbandingan (fix memori, `cycle`) memakai **2–5 seed** dan disebutkan.
  Variasi antar-seed besar; jangan menyimpulkan dari satu seed.

---

## 1. MASALAH YANG HARUS DI-FIX (berurut prioritas)

### P0 — Kualitas teks: kalah dari baseline adil

| Metrik | TMT2 | Baseline adil | Selisih |
|---|---|---|---|
| WikiText penuh, token BPE | 2,588 | token 1-gram **2,115** (tokenizer dilatih pada prefiks 20%) | **kalah 0,47 bit** |
| Lisensi BPE V=1024 | 2,757 | token bigram **2,401** | kalah ~0,58 bit |
| WikiText penuh, byte | 2,904 (d=32) | byte n-gram 2 = 2,724 | kalah 0,18 bit |

Ini masalah **utama**. Semua yang lain sekunder.

**Bukti bahwa hambatannya di luar memori:** memperbaiki jalur baca memori sampai kondisi oracle
(`Wq=Wk`, `Wk=Wv=I`, `E` ortogonal, `beta=1,0`) menghasilkan paruh2 **2,55** pada tugas salin —
tetapi pada WikiText 300k byte hasilnya **3,071** vs **3,083** tanpa memori. Perbaikan **0,0085 bit**,
konsisten antar-seed tetapi ~0,01 bit. Kontribusi marginal SSM diukur **dua metode independen**:
0,007 bit dan 0,011 bit. **Konsisten: memori bukan hambatannya.**

**Yang sudah TERBUKTI BUKAN penyebab** (jangan ulang):
- Kapasitas lebar (`d`): d=64 pada WikiText hanya 2,862 vs 2,904 di d=32; kurva **plateau**.
- Gate rekurensi: half-life multi-skala sudah memadai (L0 median 0,2; L2 median 152, maks 1717).
- `s_decay`: tidak berpengaruh pada recall.
- Replay korup: diperbaiki, efek kualitas **~0** (3,865 vs 3,794 lama).
- Kepala keluaran: kepala kosinus memang rusak dan sudah diganti kepala linear (4,970 → 3,794
  pada lisensi byte). Masih kalah setelahnya → bukan akar tunggal.
- Gradien hanya ke lapisan puncak: sudah diperbaiki `deep_grad` (4,970 → 3,794). Masih kalah.
- Memori: baru dibuktikan hari ini (lihat P1).

**Arah yang belum dijelajahi:** ekspresivitas rekurensi SSM itu sendiri dan jalur baca non-memori
(`Wc`, GLU, kepala). Karena kontribusi marginal SSM ~0,01 bit, dugaan utama adalah model
**hampir tidak menambah apa pun di atas prior n-gram**.

**→ SUDAH DIJAWAB 2026-10-04 (lihat §17).** Dugaan itu **BENAR dan kini terkuantifikasi**:
state SSM hanya membawa informasi setara **n-gram orde ~2,5**, dan SSM **redundan** begitu
mixer punya orde ≥3. Hambatannya **bukan** jalur baca (readout mempertahankan info), **bukan**
gradien, **bukan** rezim online, **bukan** smoothing n-gram. Kualitas TMT2 **= kualitas mixer**.

### P0b — Gradien terpotong (DITEMUKAN + SEBAGIAN DIPERBAIKI 2026-10-01)

Ditemukan lewat beda-hingga terpusat pada model beku (`adapter_rank=0`, `slow_every=10^9`),
mengukur gradien `L0.Wc` terhadap CE byte:

| T | 1 lapisan | 2 lapisan | 3 lapisan |
|---|---|---|---|
| 2 | 3,3e-10 | 7,6e-02 | 1,5e-01 |
| 8 | 6,1e-10 | 9,6e-01 | 1,0e+00 |
| 40 | 3,1e-09 | **6,5e+00** | **7,0e+00** |

**Tiga pemotongan terpisah, semuanya terukur:**

1. **`deep_grad` kehilangan suku gerbang — SUDAH DIPERBAIKI.** `x_l` menggerakkan gerbang `a`
   dan `u` lapisan `l`, bukan hanya jalur residual `y = x + f`. Kode lama menyetel
   `below.gh = dL/dy_l`, jadi `dL/dx_l = dL/dy_l` — kehilangan `dL/dh_l · (dh_l/dx_l)`.
   Suku itu sekarang dihitung di `flush_rec` (`o_gx`), dikendalikan `Config.deep_grad_gate`
   (default **True**).
   Terukur pada T=2: `L0.Wc` **3,11e-01 → 1,09e-10**, `L0.Wa` 4,67e-01 → 2,78e-10,
   `L0.B` 4,67e-01 → 1,86e-10. **Dalam satu langkah kini EKSAK.**

2. **Kopling TEMPORAL antar-lapisan masih hilang.** Jejak RTRL bersifat per-lapisan — ia hanya
   melacak `dh_l/d(theta_l)`. Gradien eksak ke parameter lapisan bawah lewat rekurensi lapisan
   atas memerlukan sensitivitas lintas-lapisan (O(d³)). Setelah perbaikan #1, galat pada T=6
   masih 0,46 dan pada T=20 0,62. **Belum diperbaiki.**

3. **Gerbang kejutan menambah jalur tak terlacak.** `beta = beta_max · min(1, surprise)` dengan
   `surprise` bergantung pada `pp = l2n(y_{t-1}·P)`. Satu lapisan: **7,97e-02** (gerbang aktif)
   vs **3,43e-09** (`mem_surprise_fixed=True`). **Konfigurasi terbaik sudah memakai flag itu.**

### Apakah memperbaiki gradien menaikkan kualitas? **TIDAK.**

WikiText 300k byte, d=32, byte, 2 seed:

| seed | gradien lama (`--no_gate`) | gradien diperbaiki |
|---|---|---|
| 0 | 3,115 | 3,129 |
| 1 | 3,100 | 3,108 |
| **rata-rata** | **3,1075** | **3,1185** |

Kedua seed konsisten: gradien yang **lebih benar** justru **0,011 bit lebih buruk**.
Jadi **plateau BUKAN disebabkan pemotongan gradien** — hipotesis ini gugur.

`Config.deep_grad_gate` default **True** (matematis benar) meski empiris 0,011 bit lebih buruk;
pakai `--no_gate` untuk perilaku lama. Bisa dibalik bila kualitas lebih diprioritaskan.

**Tes penanda:** `test_rtrl_one_layer_is_exact_multi_layer_is_truncated` dan
`test_surprise_gate_breaks_rtrl_exactness` (55 tes). Keduanya akan gagal bila pemotongan
diperbaiki lebih jauh — saat itu batasannya harus diperbarui.

**Catatan tentang klaim "RTRL eksak".** `test_rtrl_exact_vs_bptt` memakai `small_setup` dengan
`n_layers=1` — klaim "eksak ~5e-16" sah **hanya untuk satu lapisan**. Jangan generalisasi.

### P1 — Mekanisme memori: akar sudah ditemukan, fix belum diuji di teks

Dua penyebab, keduanya terukur:

**(a) Init `Wq` salah.** Default `Wq = N(0, 1/√d)`. Dengan `l2norm`, ini menghasilkan arah kueri
acak → `am = M q` tak berkorelasi dengan penerus → **jalur baca mati**. Ada plateau
chicken-and-egg: untuk belajar `Wq` butuh readout bagus; untuk belajar readout butuh `Wq` bagus.

Sapuan init (oracle, L=4, V=32, n=30k, lr=3e-3, se=4):

| init `Wq` | paruh2 | catatan |
|---|---|---|
| `I` (oracle) | 2,647 | ✓ |
| `I + noise 0,05` | 2,639 | ✓ (basin luas) |
| `I + noise 0,20` | 2,930 | ✓ |
| `rand` 0,05 | 5,005 | ✗ |
| `rand` 0,18 (**default model**) | 5,015 | ✗ |
| `rand` 0,5 / 1,0 / 3,0 | 5,01 | ✗ semua |
| `ortho` | 5,005 | ✗ |

Bukan soal **norma** (rand 3,0 dan ortho pun gagal) — soal **keselarasan dengan `Wk`**.

**Fix: `Config.mem_wq_match_k = True` (init `Wq = Wk`).** Terbukti:
- oracle `learn=q --wq_init match_k`: L=4 **2,553 ± 0,026**, L=8 **2,046 ± 0,032** (vs 5,013)
- oracle `learn=qkv --wq_init match_k`: **2,699** (vs 5,014)
- model nyata: 5,014 → **4,783 ± 0,071**
- **Default masih `False`** → belum aktif; perilaku lama utuh.

**(b) `d_mem` terlalu sempit.** Default `d_mem = max(8, d//4)` = 8 pada d=32.

| `d_mem` | `match_k=True` | `match_k=False` |
|---|---|---|
| 8 (default) | 4,648 | 4,976 |
| 16 | 3,986 | 5,007 |
| 32 | **3,336** | 5,014 |

Kedua fix harus bersamaan; salah satu saja tidak cukup.
**PENTING — temuan ini TIDAK berlaku di teks alami**: pada WikiText `d_mem=32` justru **merugikan**
(3,157 vs 3,071). Menambah `d_mem` menolong tugas salin, bukan bahasa alami.

**(c) Hasil uji di teks alami (WikiText 300k byte, d=32, byte, 2 seed).** Ini menutup pertanyaan
terbuka "apakah fix punya efek kualitas nyata":

| konfigurasi | seed 0 | seed 1 | rata-rata |
|---|---|---|---|
| memori **MATI** | 3,083 | 3,083 | **3,083** |
| default (memori on, `Wq` acak) | 3,118 | — | 3,118 |
| **fix, `d_mem=8`** | 3,071 | 3,078 | **3,0745** |
| fix, `d_mem=32` | 3,157 | — | 3,157 |

**Bacaan:** fix mengubah kontribusi memori dari **negatif (−0,035)** menjadi **positif kecil
(+0,0085)** — konsisten di kedua seed (kedua seed fix < kedua seed memori-mati). Terhadap
konfigurasi default yang rusak, perbaikannya 0,044 bit. **Tapi magnitudonya ~0,01 bit** — persis
seperti kontribusi marginal SSM yang diukur sebelumnya. Jadi fix ini **memperbaiki mekanisme,
bukan menyelesaikan masalah P0**: TMT2 tetap kalah ~0,47 bit dari baseline token adil.

**Yang tetap gagal (hasil negatif, jangan ulang):**
- `mem_write_grad` (Wk/Wv dapat dilatih): 4,925 vs 4,783 — **masih lebih buruk**, walau jalur baca
  sudah hidup. Jadi hasil negatif lama **bukan** artefak jalur baca yang mati.
- Kueri dari `h_t`: **5,01 di semua varian**, termasuk dengan `Wq` terlatih penuh. Klaim
  "kueri lewat `h` adalah hambatannya" **DITARIK** — oracle `none` tidak adil untuk perbandingan ini.
- `mem_query_x` (kueri dari `ln(x_t)`) default **`False`**; diperlukan agar fix (a) berguna.

### P2 — Kecepatan (sebagian dikerjakan 2026-10-01)

Throughput **steady-state** (di luar kompilasi numba), satu proses:

| d | sebelum | sesudah | naik |
|---|---|---|---|
| 32 | 1209 B/s | **2308 B/s** | 1,91× |
| 64 | 578 B/s | **1165 B/s** | **2,02×** (target 1000 tercapai) |
| 128 | 263 B/s | 379 B/s | 1,44× |

Catatan: Tugas 9 melaporkan d=32 = 2025 / d=64 = 666 / d=128 = 227 B/s (metodologi beda).
Pengukuran lebih awal di sesi ini melaporkan 780 B/s, tetapi itu dengan **dua proses paralel**
— setengah dari throughput sebenarnya. **Selalu ukur satu proses.**

**Optimasi yang dikerjakan (semua tetap eksak, 52 tes lolos):**
1. `l2n_np` jalur 1-D: `x @ x` (2 operasi numpy) alih-alih `(x*x).sum()` (4 operasi).
   Fungsi ini dipanggil ~6×/byte.
2. `_resolve`: `D = l2n_np(m.dict_tensor().data)` dijadikan **malas** — dengan kepala linear
   nilai ini tidak pernah dipakai, padahal menormalisasi kamus keluaran penuh setiap byte.
3. `_consume`: `_weight` diangkat keluar loop, `np.linalg.norm` diganti dot product
   (overhead Python `np.linalg.norm` besar untuk array kecil).
4. **`_mv`/`_vm`/`_dot` di `fastpath.py` — pengaruh terbesar.** Bentuk broadcast
   `(A * v[None,:]).sum(1)` mengalokasikan dan menulis **temporer penuh (m,n)** setiap panggilan
   (256×32 = 64 KB untuk kepala keluaran) lalu mereduksinya — dua lintasan. Diganti loop
   eksplisit: **7,2× lebih cepat** pada (256,32). `head_fwd_lin` turun dari 20,3 → 7,3 us.
   Beda numerik ~3e-15 (orde epsilon float64).

**Analisis lantai (PENGUKURAN ULANG, revisi 6).** Pisah waktu presisi (d=32/2L, 254 us/byte):
**112 us DI DALAM njit (40,7%) vs 163 us Python murni (59,3%)**; 8,12 panggilan njit/byte.

| kernel | us | jumlah |
|---|---|---|
| `step_fwd` (1 lapisan, 42 arg) | 10,5 | ×2 |
| `flush_rec` (1 lapisan, 43 arg) | 13,5 | ×2 |
| kepala (`head_fwd_lin`+`head_bwd_lin_acc`+`softmax`+`ce_val`) | 18,5 | ×1 |
| `adamw_step` (siklus /8) | 152 | /8 |

→ **lantai njit ≈ 53 us/byte = batas atas ≈ 19.000 B/s** bila SELURUH overhead Python hilang.
Sekarang 254 us/byte → **~200 us masih Python murni**.

**Sumber overhead Python = ARITY kernel.** Mikro: noop 1-arg 0,23 us vs noop 30-arg **2,53 us**.
`step_fwd`/`flush_rec` = 42/43 arg → ~4,2 us Python/panggilan x 8 = **~20 us/byte hanya dispatch**.

**Batas kernel TERPADU — DIUKUR ULANG, KLAIM 38.540 B/s DITARIK (revisi 7).** Prototipe
`scripts/proto_fused_est.py` TIDAK pernah mengukur sebyte penuh (tanpa mundur, jejak lapis-2
parsial, tanpa memori). Setelah komponen hilang ditambahkan: **39,98 us/byte**, bukan 25,95.
Kernel terpadu kini NYATA (`fused_fwd`/`fused_bwd`, bit-identik, 77 tes lolos). End-to-end
nyata (`bench_fused_e2e.py`, korpus license, 8k byte, `slow_every=256`):
**lama 3.298 → fused 8.054 B/s @ d=32 (2,44x)**; 1.673 → 2.790 B/s @ d=64.
Plafon jujur ≈ **10–12k B/s @ d=32** (`proto_lower_bound.py`: maju saja 20,85 us/lapisan).

**Titik sasaran B/s:**

| tujuan | data | B/s dibutuhkan | status |
|---|---|---|---|
| Ablation (10 run × 300k byte) | 3 MB | ~1250 | tercapai |
| 1 sesi, 50 MB | 50 MB | ~1390 | tercapai |
| **target user, 5k B/s** | — | 5000 | **tercapai (fused 8.054)** |
| **target user, 20k B/s** | — | 20000 | **TIDAK tercapai** (8.054; butuh 2,5x lagi) |
| train penuh | 2,23 GB | ~51600 | butuh GPU |

**Kenapa 20k tidak realistis** (bukan soal tuning): mundur didominasi jejak RTRL `3·d²`
= 12,98 us/lapisan @d=32 (26,5 @d=64), ~55% waktu mundur, irreducible tanpa BLAS; loop `mm`
6–10x lebih lambat dari BLAS (`proto_dgemm.py`, `proto_trace_cost.py`). Untuk menembusnya
perlu (a) `d` lebih kecil (d=16 ≈4x lebih murah), (b) jejak RTRL low-rank/subspace, atau
(c) BLAS via scipy (**DILARANG** aturan proyek). **Keputusan: fokus pindah ke kualitas (BPB).**

**Optimasi yang SUDAH mendarat** (revisi 6): `TrainOpts.use_njit_adam=True` (slow_update:
820 → 152 us = 5,4x, setara 4,4e-16), `norm_fast=True`. Hasil end-to-end HONEST (bench adil,
warmup 500, best-of-5): **1,09x @ d=32 / 1,29x @ d=64 / 1,32x @ d=128**. `slow_update`
63 → 31 us/byte. **Catatan metodologi**: benchmark yang MEMAKAI ULANG satu model antar-rep
menghasilkan angka terlalu tinggi (3.935) karena state sudah konvergen; angka adil memakai
model+warmup identik per pembanding.

**Kesimpulan: kecepatan BUKAN hambatan kualitas saat ini.** Untuk eksperimen yang kita perlukan
(ablation, 10–50 MB), throughput sudah memadai. Kecepatan hanya jadi hambatan kalau terbukti
lebih banyak data menolong — dan bukti yang ada justru menunjukkan **kurva plateau**
(WikiText ~500k byte; d=64 hampir tak beda dari d=32).

### P3 — Skala korpus & infrastruktur

- Korpus lisensi hanya **46.507 byte** → efek 0,01 bit tak terukur di situ. Butuh korpus lebih besar.
- WikiText penuh 1.121.681 byte; kurva **plateau** setelah ~500k byte.
- Jalur BPE: 302.599 token, 25 menit/run. Kepala V=4096 jadi hambatan kecepatan.
- `README.md` **kedaluwarsa**: menyebut "29 tes" (sekarang 52) dan belum menyebut `tokenizer.py`,
  `corpus.py`, `fastpath.py` terbaru, `deep_grad`, `head=linear`, `mem_query_x`, `mem_wq_match_k`.

---

## 2. KESIMPULAN TERVERIFIKASI

### 2.1 Ketepatan mekanisme (fondasi kokoh)

- **RTRL diagonal eksak vs BPTT ~5e-16.** `autograd` adalah definisi kebenaran.
- Jalur cepat numba == `Learner` sampai **~4e-15**; setelah Tugas 9 tetap eksak **3e-16**.
- `deep_grad`, kepala linear, `mem_query_x`, `mem_hetero`: semua divalidasi eksak + gradien
  (`Wq` rel err 3,6e-15 / 5,6e-15). `mem_write_bwd` tervalidasi 2,3e-09.
- Kepala linear: eksak 1,8e-15; `l2norm(p)` **wajib** — tanpanya divergen (BPB 1074).

### 2.2 Hasil kualitas (jujur, apa adanya)

- Konteks deterministik **berhasil**: periodik 0,414 bit (unigram 4,397); sintetis 0,983 bit.
- Lisensi byte: 4,970 → 4,924 (`head=linear`) → **3,794** (`deep_grad`) → 3,865 (P0b).
- WikiText byte: d=32 **2,904** (1/3 akhir 2,823); d=64 **2,862** (2,762). Baseline byte:
  unigram 4,610 / n-gram 1 = 3,374 / n-gram 2 = 2,724.
- WikiText BPE: **2,588**. Tapi baseline pada token yang sama: unigram **2,551**, bigram **2,134**
  → **TMT2 kalah**. Klaim "melewati n-gram 2" **hanya sah terhadap baseline BYTE** (lebih lemah).
- Baseline adil (tokenizer hanya pada prefiks 20%): token 1-gram **2,115** → TMT2 kalah **0,47 bit**.

### 2.3 Garis memori (M) — rangkuman lengkap

- Memori delta-rule: `M += β·outer(v − M·k, k)`; baca `q = l2n(ln(h)@Wq)`, `am = M q`.
- **Auto-asosiatif**: `k` dan `v` sama-sama dari `x_t` → menyimpan token ke dirinya sendiri.
  `mem_hetero=True` (kunci dari `ln(x_{t-1})`) memperbaikinya; tes unit cos −0,55 → +0,91.
- **Gerbang kejutan MERUGIKAN**: `mem_surprise_fixed=True` (`β = β_max`). Dengan `β_max=1,0`
  oracle turun besar: L=4 3,711 → **2,598**, L=8 3,121 → **1,890**. Sisa celah ke lantai (~1,3 bit)
  bukan dari `β`.
- **Penulisan beku bukan penghambat** (dikonfirmasi ulang hari ini).
- Kecepatan `mem_hetero` kini **identik** dengan auto (1,00x; 1457 vs 1453 tok/s) setelah cabang
  hetero dipindah keluar loop dalam.

### 2.4 Temuan hari ini (2026-10-01)

1. **Kontrol `cycle=True` LULUS di semua regime, bahkan d=4** → kepala + embedding **mampu**
   belajar pemetaan token; anggaran optimasi memadai untuk tugas tanpa memori jarak jauh.
   `cycle` adalah kontrol untuk **kepala + embedding saja**, BUKAN untuk optimasi jalur memori.
2. **`mem_query_x`**: kueri dari `ln(x_t)`. Eksak 4,4e-15, gradien `Wq` 5,6e-15.
3. **Oracle**: `x_t` turun monoton (4,50 → 4,05 → 3,71 → 3,12 → 2,66) sedangkan `h_t` **datar
   tepat di 5,0**. Pipeline baca (`Wm`/GLU/kepala) **terbukti bisa** memakai `am`.
   *Kejujuran*: dengan `Wq=I`, `ln(x_t)@Wq = ln(x_t)` persis → itu **oracle**, bukan model terlatih.
4. **Akar init `Wq` + `d_mem`** (lihat P1) — temuan paling penting hari ini.

### 2.5 Metodologi: lantai yang benar (koreksi)

**"Ideal 0 bit" pada `copy_task(perm=True)` SALAH.** Pada paruh 2, posisi pertama blok kedua
(`blk[0]`) diprediksi dari `blk[L-1]`, dan penerus `blk[0]` **belum pernah terlihat** → tidak bisa
dipanggil dari memori. Biaya ~`log2(V)` untuk 1 dari tiap `L` posisi:

| L | lantai ≈ log2(32)/L |
|---|---|
| 2 | 2,5 |
| 4 | 1,25 |
| 8 | 0,63 |
| 16 | 0,31 |

Lantai ini **KASAR** (smoothing bisa menebak sebagian). Untuk V=4 non-perm, lantainya lebih rumit
karena ambiguitas kunci berulang. **Semua pembacaan probe salin sebelumnya harus dikoreksi
terhadap lantai ini.**

**Jebakan lain yang sudah ditemukan:**
- `V=4` + `L≥4` → kunci 1-token berulang → "hafalan" tak terbedakan dari "prediksi 1 langkah".
  Diperbaiki dengan `perm=True` (butuh `L ≤ V`).
- Membandingkan angka dari **regime berbeda** (n, lr) tidak sah — pernah terjadi, sudah dihindari.
- Uji gradien vs beda-hingga **wajib** `adapter_rank=0` + `slow_every` besar.
- `E` tidak sebanding antara `Learner` dan `FastLearner` (cakupan langkah berbeda) — bukan bug.
- Benchmark bising; pakai aliran ≥10k byte dan rasio, bukan angka sekali jalan.

---

## 3. YANG DITARIK / HASIL NEGATIF (jangan diulang)

| Klaim lama | Status |
|---|---|
| "Regime default probe terlalu tumpul" | **DITARIK.** `cycle` lulus di regime default; `lr=1e-3/se4` bukan "lebih baik", hanya trade-off lain. |
| "Kueri lewat `h` adalah hambatannya" | **DITARIK.** Oracle `none` tidak adil (identitas atas campuran berpeluang gagal). Dengan `Wq` terlatih, `h_t` tetap 5,01. |
| "Ideal paruh2 = 0 bit" | **SALAH.** Lantai = `log2(V)/L`. |
| "TMT2 melewati n-gram 2" | **Hanya** terhadap baseline byte; kalah dari baseline token. |
| "Penulisan memori dapat dipelajari akan menolong" | **NEGATIF**, dikonfirmasi ulang (4,925 vs 4,783). |
| "`s_decay` adalah hambatan" | **GUGUR.** |
| "Replay korup menyebabkan plateau" | **GUGUR** (efek kualitas ~0). |
| "Tes unit cos −0,55 → +0,91" | **TIDAK ADA di repo.** Diganti `test_mem_query_x_retrieval_through_pipeline`. |
| "Mixer orde 1..8 menurunkan BPB beku 0,83 bit / 1,138 bit" | **DITARIK** (§16) — kolom beku tercemar kebocoran jendela uji ke `_mix_ng`. Efek orde jujur: beku **1,35 bit** (3,49→2,14, lalu jenuh) / online **1,45 bit** (3,45→2,00), sapuan bersih 60k/8k. |

> **Catatan §4 item 1 & 3 (jalur non-memori)**: kesimpulan "hambatan di jalur non-memori" **tetap
> berlaku**, tetapi sekarang diperkuat oleh mixer (§12–§14): jalur **konteks-lokal eksplisit**
> (mixer) jauh mengalahkan SSM-saja (~0,7–1,0 bit), dan kontribusi marginal SSM ~0,01 bit.
> Angka absolut mixer yang dulu dikutip (1,138) dikoreksi di §16, tetapi **arah** kesimpulan
> (SSM bukan sumber kualitas) **tidak berubah** karena bertumpu pada kolom online yang bersih.

---

## 4. RENCANA BERIKUTNYA (berurut)

1. ~~Uji `mem_wq_match_k=True` + `d_mem=d` pada WikiText penuh.~~ **SELESAI (2026-10-01).**
   Hasil: fix menolong **0,0085 bit** (konsisten antar-seed) tetapi ~0,01 bit; `d_mem=32`
   merugikan. **Kesimpulan tegas: hambatan TMT2 ada di jalur NON-MEMORI.**
2. **Pertimbangkan menjadikan `mem_wq_match_k=True` + `mem_query_x=True` default** — sekarang
   terbukti menolong (kecil) di teks alami, bukan hanya di oracle. Uji dulu dengan 3 seed penuh.
3. **Selidiki jalur non-memori** — sekarang ini **prioritas utama P0**: `Wc`, GLU, kepala, dan
   ekspresivitas rekurensi SSM. Karena kontribusi marginal SSM ~0,01 bit, pertanyaan sebenarnya
   adalah *apa yang model pelajari di atas prior n-gram*.
4. Baru setelah model mengalahkan baseline token adil: sweep `(lr, slow_every)`, BPTT untuk `M`,
   dan P2 (kecepatan).
5. Perbarui `README.md` yang kedaluwarsa.

---

## 5. PETA ARTEFAK

| Berkas | Isi |
|---|---|
| `tmt2/model.py` | `Config` + `TMT2`. Flag penting: `head="linear"`, `deep_grad`, `mem_query_x`, `mem_wq_match_k`, `mem_hetero`, `mem_surprise_fixed`, `beta_max`, `mem_write_grad` |
| `tmt2/core.py` | acuan autograd: `layer_step`, `ff_readout`, RTRL |
| `tmt2/fastpath.py` | kernel numba: `step_fwd`, `flush_rec`, `mem_write_bwd`, `head_*_lin` |
| `tmt2/fast_learner.py` | `FastLearner`, arena datar, AdamW in-place, `only_params` |
| `tmt2/learner.py` | `TrainOpts`, `Stream`, sinyal belajar |
| `tmt2/tokenizer.py` | BPE, akuntansi byte eksak |
| `tmt2/corpus.py` | loader + `copy_task(perm=, cycle=)` |
| `scripts/bigtest.py` | pelatihan korpus (`--bpe`, `--ngram_*`) |
| `scripts/fair_baseline.py` | baseline adil (tokenizer pada prefiks) |
| `scripts/probe_state_info.py` | probe info-state (§17.1): info byte-berikutnya di state vs readout |
| `scripts/probe_epochs.py` | probe epoch (§17.2) + ablasi mixer (`--mixer/--zero_out/--match`) |
| `scripts/ngram_ceiling.py` | lantai n-gram byte JM orde 1..8 (§17.3) |
| `scripts/mixer_variants.py` | skema interpolasi mixer vs WB-penuh vs aritmetik (§17.3) |
| `scripts/kn_ceiling.py` | lantai Kneser-Ney byte (§17.3) |
| `scripts/fair_online_compare.py` | papan skor ADIL held-out (§18.0): TMT2 vs n-gram/ctx-mix/bz2 kondisional |
| `scripts/probe_readout_capacity.py` | probe linear vs GLU vs MLP pada state (§17.1) |
| `scripts/fair_bpe_split.py` | baseline BPE token pada split yang sama (§18.1) |
| `scripts/gate1_sweep.py` | GATE 1 penuh: 3 seed × 2 korpus, PARALEL (multiprocessing) (§18.1) |
| `scripts/ngram_order_sweep.py` | kontribusi marginal SSM per orde |
| `scripts/memory_probe.py` | probe salin; `--perm`, `--cycle`, `--trace`, `--mem_query_x`, `--mem_wq_match_k` |
| `scripts/oracle_probe.py` | oracle; `--learn {none,q,qkv,all}`, `--wk_init`, `--wq_init`, `--beta_max` |
| `checkpoints/` | `wikitext_full.npz`, `wikitext_d64.npz`, `license.npz`, dll. |

**Konfigurasi terbaik saat ini (byte):** `head="linear"` + `deep_grad=True` +
`TrainOpts(only_byte_loss=True)` + `lr=1e-2` + `warmup=200` + **mixer orde 1..12 + `mixer_match`**.
Loss auxiliary (JEPA/VICReg/stop/state) justru **merugi** (4,925 vs 3,794).

---

## 17. KUALITAS — DI MANA SEBENARNYA HAMBATANNYA (2026-10-04)

Menjawab §1 "arah yang belum dijelajahi". Semua angka WikiText, d=32, L=2, byte, protokol
prequensial, CE tahan (state dingin). Skrip baru: `scripts/probe_state_info.py`,
`probe_epochs.py`, `ngram_ceiling.py`, `mixer_variants.py`, `kn_ceiling.py`.

### 17.1 Probe info-state: state SSM ≈ n-gram orde 2,5
Latih 200k (fused), tahan 10k. Fitur → kepala linear; **dilatih pada 70% bagian tahan, dinilai
pada 30% sisanya**. N-gram empiris dari korpus LATIH, diuji out-of-sample (tidak terpengaruh
perbaikan probe).

| representasi | bit | | baseline n-gram | bit |
|---|---|---|---|---|
| E[b_t] (byte sekarang) | 3,658 | | order-1 | 3,703 |
| h0 (state lapisan 0) | 3,363 | | order-2 | 3,049 |
| h1 (state lapisan 1) | 3,536 | | order-3 | 2,623 |
| q1 = masukan kepala | 3,203 | | | |
| **model sendiri (CE)** | **2,974** | | | |

⚠️ **KOREKSI 2026-10-04.** Versi pertama probe ini mengevaluasi pada himpunan LATIH, sehingga
angkanya CE LATIH (E 3,224 · h0 2,788 · q1 2,738) — **DITARIK**; di atas adalah CE UJI.
**Rekurensi menambah +0,295 bit di atas byte sekarang; jalur baca menambah +0,334 bit di atas
state mentah** → keduanya BEKERJA (bukan hambatan).

**Yang tetap teguh**: CE model sendiri **2,974 jatuh di antara order-2 (3,049) dan order-3
(2,623)** → model SSM-saja ≈ **n-gram orde-2,5**. Perbandingan ini memakai CE-model vs CE-n-gram
(keduanya terlatih penuh), jadi bebas dari kelemahan probe. Konsisten dengan ablasi mixer §17.4.

**Kapasitas readout BUKAN penghambat** (`probe_readout_capacity.py`, CE uji): pada state —
linear 3,536 / **glu 3,849 (lebih buruk)** / mlp 3,478; pada masukan kepala — linear **3,203** /
glu 3,358 / mlp 3,280. Readout berkapasitas lebih tinggi **tidak** mengambil info tambahan →
yang membatasi = **KANDUNGAN STATE**.

*Catatan kejujuran*: probe hanya punya 6.999 sampel latih (vs 200k byte milik model), jadi angka
absolut probe lebih PESIMISTIS dan **tidak boleh dibandingkan langsung dengan CE model**
(2,974 < 3,203). Yang sah adalah perbandingan RELATIF antar-fitur / antar-kapasitas.

### 17.2 Probe epoch: rezim online BUKAN penghambat
WikiText 100k, 5 epoch: 3,034 → 3,001 → 2,950 → 2,923 → 2,928. **Hanya +0,11 bit**, jenuh.
Kapasitas/arsitektur yang membatasi, bukan anggaran lintasan.

### 17.3 Lantai n-gram byte (200k latih / 8k uji, out-of-sample)
| order | 1 | 2 | 3 | 4 | **5** | 8 |
|---|---|---|---|---|---|---|
| JM | 3,345 | 2,632 | 2,171 | 1,981 | **1,925** | 1,960 |

Kneser-Ney (D=0,75) terbaik 1,947 — **KN tidak mengalahkan JM/WB** di byte V=256 →
**ganti smoothing bukan tuas**.

### 17.4 Ablasi (200k, seed 0): SSM menyumbang NOL di atas mixer
| konfigurasi | bit |
|---|---|
| SSM saja | 2,9235 |
| **mixer saja** | **1,8434** |
| mixer + SSM | 1,8440 |

Dan SSM hanya mengisi konteks **orde rendah** (100k): mixer orde-1 tanpa SSM 3,0370 vs +SSM
2,9707 (**+0,066**); orde-2 2,7106 vs 2,7018 (+0,009); orde-8 2,1632 vs 2,1632 (**0,000**).
→ **SSM ≈ model konteks order-2**; redundan begitu mixer punya orde ≥3.

### 17.5 Kualitas TMT2 = kualitas mixer — dan itu MENANG atas baseline
Model+mixer **1,8264 bit** (200k/8k, 3 seed, mixer orde 1..12 + match) **mengalahkan**
lantai byte 5-gram (1,925) dan BPE unigram adil (2,115). Yang kalah baseline adalah
**jalur SSM-saja** (2,92), bukan sistem lengkap.

### 17.6 Perbaikan yang MENDARAT
`Config.best()` kini memakai **mixer orde 1..12 + `mixer_match=True`**. Terukur 3/3 seed:
1,8420 → **1,8264 bit** (−0,0156). 77 tes tetap lolos.

### 17.7 Yang TIDAK menolong (jangan ulang)
KN smoothing (1,947 vs JM 1,925) · multi-epoch (+0,11 bit) · `mixer_lr_mult` 1,0 (−0,004) ·
`mixer_init` 0,3 (−0,011) · memperbaiki readout (probe: readout sudah mempertahankan info).

### 17.8 Untuk naik lebih jauh
Rekurensi diagonal SSM **secara struktural** hanya filter-bank low-pass (transisi diagonal,
tanpa pencampuran antar-saluran) → tidak bisa merepresentasikan fitur lokal tajam orde tinggi.
Perlu salah satu: (a) mixer lebih kuat (keuntungan kecil, ~0,02-0,03 bit), (b) interpolasi
aritmetik mixer⊕SSM (offline 1,809; ~0,03-0,08 bit), atau (c) rekurensi dengan pencampuran
antar-saluran (memecah trik RTRL diagonal — perubahan besar).

---

## 18. JALAN KE PRODUKSI — gate yang harus dilewati (2026-10-04)

**Posisi sekarang: Tahap 0 (sandbox mekanisme).** Verdict §0 belum berubah. Lima gate
berurutan; **Gate 2 memblokir seluruh rantai** — selama SSM tidak mengalahkan n-gram, tidak
ada alasan produksi.

### 18.0 Papan skor adil saat ini (`scripts/fair_online_compare.py`)
Semua "dilatih" pada `data[:200000]` WikiText, dinilai pada `data[200000:208000]` yang TAK
PERNAH dilihat; baseline dibekukan setelah fase latih.

| metode | bit/byte |
|---|---|
| n-gram JM orde-5 (beku) | 1,9254 |
| **bz2 -9 KONDISIONAL (tr\|te)** | **1,9450** |
| ctx-mix orde-4 (beku, diskon absolut) | 2,0186 |
| lzma -9 KONDISIONAL (tr\|te) | 2,1480 |
| **BPE token 1-gram** (V=4096, tokenizer dilatih pada tr) | **2,2129** |
| zlib -9 KONDISIONAL (tr\|te) | 2,5630 |
| **TMT2+mixer (state HANGAT)** | **1,8273** |
| TMT2+mixer (state DINGIN) | 1,8279 |

Pada **WikiText**, TMT2+mixer mengalahkan SEMUA baseline adil — termasuk bz2 -9 (target lama
2,2496) dan BPE token 1-gram. **Pada korpus Indonesia ia KALAH dari bz2 -9 & lzma -9** (lihat
§18.1). Kreditnya milik **mixer**, bukan SSM (kontribusi SSM 0,000 bit).
Referensi protokol ONLINE (seluruh 200k, dari `strong_baseline.py`, `results/strong_baseline_fixed.jsonl`):
bz2 -9 2,2496 · lzma -9 2,4406 · zlib -9 2,7570 · ctx-mix-3 2,3870 · ctx-mix-4 2,2795 · match 3,0123.
**Angka online vs beku vs kondisional TIDAK boleh dibandingkan langsung** (jendela berbeda).
Skrip: `fair_online_compare.py` (byte), `fair_bpe_split.py` (token BPE, split sama).

### 18.1 Gate 1 — Baseline adil · **BELUM LULUS** (3 seed × 2 korpus)
`scripts/gate1_sweep.py` (paralel, 156 detik untuk seluruh sweep):

| baseline | WikiText | Indonesia |
|---|---|---|
| **TMT2+mixer** | **1,8264 ± 0,0014** | **2,4809 ± 0,0068** |
| n-gram JM orde-5 | 1,9254 · menang | 2,5400 · menang |
| ctx-mix orde-4 | 2,0186 · menang | 2,8491 · menang |
| **bz2 -9 kondisional** | 1,9450 · menang | **2,3700 · KALAH (+0,111)** |
| **lzma -9 kondisional** | 2,1480 · menang | **2,4040 · KALAH (+0,077)** |
| zlib -9 kondisional | 2,5630 · menang | 2,8060 · menang |
| BPE token terbaik | 2,2129 · menang | 3,0050 · menang |

**Menang di semua baseline pada WikiText, KALAH dari bz2 -9 & lzma -9 pada Indonesia.**
Sensitivitas jendela diuji (offset 200k/400k/600k/800k): bz2 = 2,370/2,093/2,336/2,206 → kalah di
SEMUA jendela, bukan artefak satu jendela. **Tafsir**: Indonesia ensiklopedis & sangat repetitif;
LZ77 bz2 (jendela 900 KB) menangkap ulangan jarak-jauh, mixer hanya orde ≤12 byte. **Korpus yang
punya struktur jarak-jauh justru tempat jalur memori/SSM seharusnya bernilai — dan di situ gagal.**
Kredit pada WikiText tetap milik **mixer**; SSM 0,000 bit.

### 18.2 Gate 2 — Nilai SSM · **MEMBLOKIR**
Sekarang **0,000 bit** di atas mixer (1,8440 vs 1,8434) dan **melemah** saat mixer menguat:
+0,066 @orde-1 → +0,009 @orde-2 → 0,000 @orde-8; audit lama bahkan −0,057 bit di 500k.
- **Ambang usulan**: SSM ≥ **+0,05 bit** di atas mixer pada ≥1M byte, dan **tidak mengecil**
  saat data bertambah.
- **Kalau gagal**: keputusan produk yang jujur = pakai mixer saja, buang SSM (membatalkan
  tesis proyek, tetapi itu kesimpulan yang sah).
- **Jalan lulus**: rekurensi dengan pencampuran antar-saluran (memecah trik RTRL diagonal),
  atau temukan tugas di mana SSM benar-benar mengalahkan n-gram orde-8.

### 18.3 Gate 3 — Skala data
Kurva **1M → 10M → 100M byte** monoton turun, tanpa plateau. Sekarang SSM-saja plateau ~500k;
mixer masih turun (100k 2,16 → 200k 1,84) → sinyal positif. Butuh korpus ≥100 MB; sekarang
maks **1,1 MB** (`data/wikitext_valid.txt`). TinyStories 2,23 GB / enwik8 100 MB belum diunduh.
Waktu pada kecepatan sekarang (8.054 B/s): 100M byte ≈ 3,4 jam · 1B ≈ 34,5 jam.

### 18.4 Gate 4 — Benchmark standar
**enwik8** (100 MB, split 90M/5M/5M), **text8**, **WikiText-103**. Pembanding byte-level
publik ada di kisaran ~0,95-1,1 bit/byte — **verifikasi leaderboard terkini saat menjalankan**.
Angka 1,827 sekarang **belum bisa dibandingkan** (korpus & skala beda).

### 18.5 Gate 5 — Pengerasan produksi
- **Throughput**: 8k B/s. Melatih 10B byte dalam sehari butuh ~116k B/s (≈14×) → **perlu
  GPU/BLAS**; artinya aturan no-scipy harus dilonggarkan. Alternatif: batasi klaim ke model kecil.
- **Stabilitas numerik**: kepala **wajib** `l2norm` (tanpa itu divergen) → indikasi conditioning
  buruk yang harus dihilangkan sebelum produksi.
- **Determinisme + checkpoint/resume**: `--resume`/`--save_every` ada; perlu uji bit-identik
  lintas-mesin.
- **Memori**: tabel mixer orde-12 tumbuh linear dengan data → perlu pemangkasan/backoff.
- **Serving + observabilitas**: generasi ada tapi lambat; perlu batching & pemantauan BPB.

### 18.6 Urutan kerja yang benar
**Gate 1 belum lulus** — kekalahan pada korpus Indonesia menunjuk langsung ke kelemahan
jarak-jauh. **Gate 2 (nilai SSM/memori) dan Gate 1 kini menyatu**: keduanya butuh model
menangkap struktur jarak-jauh. Selanjutnya: Gate 2 → Gate 1 (ulang) → Gate 3 (skala) →
Gate 4 (benchmark) → Gate 5 (produksi). Gate 3 sudah menunjukkan sinyal positif.

---

## 19. TMT2 v2 — TINGKAT A (inti statistik biner), 2026-10-04

Rancangan: `ARCHITECTURE_V2.md` (M0–M6). Yang **sudah diimplementasikan**: **M0** (alat ukur)
dan **Tingkat A** (`tmt2/stat_core.py`, M1–M3 sebagian). Tingkat B neural **belum ada**.

### 19.1 M0 — KOREKSI KECEPATAN PRODUKSI (penting)
| jalur | B/s |
|---|---|
| **PRODUKSI `Config.best()` (mixer ON)** | **2.666** |
| jalur fused, **mixer MATI** | 12.907 |
| **Tingkat A v2** (tabel 2²⁵) | **~100.000** |

Angka **8.054 B/s** yang dikutip sejak revisi 7 berasal dari jalur fused **tanpa mixer** —
konfigurasi yang **1 bit/byte lebih buruk**. Konfigurasi produksi berjalan **2.666 B/s**.
Dampak skala: 100M byte = 10,4 jam (produksi) vs **~17 menit** (v2).

**Optimasi kecepatan Tingkat A (50k → 100k B/s), semuanya bit-identik:**
hash konteks diangkat keluar loop bit (50k→75k) · tabel dipadatkan jadi satu int64
(3 akses memori → 1; 75k→91k) · simpan `key` saja, bukan slot+checksum (91k→~100k).
Konsekuensi: 8 B/slot → tabel 2²⁵ = **268 MB**.

### 19.2 Hasil Tingkat A (protokol G1: latih 200k / uji 8k, 1 inti)
| sistem | WikiText | Indonesia | B/s |
|---|---|---|---|
| **Tingkat A v2** (Mixer-2 + kata, tabel 2²⁵) | **1,7837** | **2,3261** | **~100k** |
| Tingkat A, tabel 2²⁶ (memori 537 MB) | 1,7784 | 2,3176 | ~98k |
| TMT2+mixer simbol | 1,8264 | 2,4809 | 2.666 |
| n-gram JM-5 beku | 1,9254 | 2,5400 | — |
| bz2 -9 kondisional | 1,9450 | **2,3700** | — |

**Tingkat A menang atas mixer simbol di KEDUA korpus** (+0,043 / +0,155) dan **mengalahkan
bz2 -9 di Indonesia** (2,3261 vs 2,3700, **+0,044**).

Ablasi dari konfigurasi final (WikiText): tabel 2²²→2²⁶ **−0,16** · APM −0,034 ·
Mixer-2 berlapis −0,012 · **konteks kata −0,012** · pemilih bobot 8→12 bit −0,023 ·
model match −0,008 · order-0 −0,003 · **konteks jarang +0,038 (MERUGI, jangan dipakai)**.

> **Metodologi kecepatan**: mesin dipakai bersama → B/s berayun ~1,5× antar-jam; laporkan
> best-of-N. Angka tunggal jangan dikutip sendirian.

### 19.3 Status gate
| gate | status |
|---|---|
| **G0** alat ukur | **LULUS** (§19.1) |
| **G1** Tingkat A | **2 dari 3**: kecepatan **LULUS** (~100k ≥ 100k) · Indonesia **LULUS** (+0,155 atas mixer simbol, **+0,044 atas bz2**) · WikiText **+0,043** (butuh +0,05 — meleset 0,007) |
| **G2** nilai neural | **BELUM LULUS** (§19.6): Tingkat B +0,008 @30k → −0,016 @100k. Dikeluarkan dari jalur produksi. |
| **G3** kecepatan | **sebagian** (§19.8): agregat 8 inti ≈ 143k B/s (4,62×) — belum ≥500k |
| **G4** skala | **LULUS** (§19.7) pada 100k→1,1M (monoton, tanpa plateau); 10M/100M belum (korpus lokal maks 1,12 MB) |
| G5 paritas | sebagian (88 tes lolos; kepala pohon biner & shard belum ditutup) |

### 19.4 Belum dikerjakan dari rancangan
Tabel mode "additive" untuk shard · Tingkat B neural (Saron/Gong/Bonang, M4–M5) ·
shard multi-inti (M6) · sisa 0,007 bit WikiText.
Tes paritas: `tests/test_stat_core.py` (6 tes; total suite **83 lolos**).

### 19.5 Tingkat B — pelengkap neural paralel-chunk (2026-10-04)
Paket `tmt2/gamelan/` (acuan numpy float64): `gong.py` (rekurensi linear peluruhan
bergantung-data, **dua bentuk**), `saron.py` (konvolusi kausal + state antar-chunk),
`block.py` (Saron + Gong + gerbang per head + SwiGLU + RMSNorm).

**Tes paritas** (`tests/test_gamelan.py`):
| tes | syarat | hasil |
|---|---|---|
| #1 Gong chunkwise == rekuren | < 1e-10 | **9,15e-14** (λ tetap/acak/kecil/≈1; C = 1…256) |
| #2 Saron state antar-chunk == deret penuh | eksak | **6,66e-16** |
| #3 Blok paralel == decode sekuensial | < 1e-9 | **lulus** (state akhir sama) |
| #4 gradien vs beda-hingga | rel < 1e-6 | **BELUM** (belum ada jalur mundur) |

**Kecepatan** (satu blok, d=32, H=4, T=2048, acuan numpy): **chunkwise C=64 = 14,8 µs/byte
(67.628 B/s)** vs rekuren 57,2 µs/byte (17.494 B/s) → **~4× lebih cepat**. Ini acuan Python,
bukan jalur cepat; versi tervektorisasi penuh akan jauh lebih cepat.

> **Jebakan**: versi pertama `gong_chunkwise` memakai loop O(C²) → **12× LEBIH LAMBAT** dari
> rekuren. Dualitas chunkwise hanya berguna bila intra-chunk benar-benar menjadi matmul;
> **paritas saja tidak cukup, kecepatan juga wajib diukur**.

**Belum ada**: Bonang · pelatihan BPTT per-chunk · kepala pohon biner (Tingkat C) · jalur mundur
+ gradcheck #4 · integrasi A+B+C (M5). Total suite **87 tes lolos**.

### 19.6 M5 — integrasi A+B dan PUTUSAN G2: **BELUM LULUS** (2026-10-04)
Jalur mundur Tingkat B (`tmt2/gamelan/train.py`) selesai dan **lulus gradcheck #4** (rel
**2,23e-07**). Integrasi: keluaran Tingkat B → `stretch(p(bit|prefiks))` lewat pohon biner →
**satu masukan mixer** Tingkat A (`StatCore.step(..., nn_st=)`). Skrip: `scripts/m5_integrate.py`.

| skala | Tingkat B mandiri | A saja | A + B | Δ |
|---|---|---|---|---|
| 30k byte, 3 epoch | 2,83 | 3,0716 | 3,0632 | **+0,0084** |
| 100k byte, 5 epoch | — | 2,9754 | 2,9909 | **−0,0155** |
| **kontrol: B ACAK** | — | 3,4503 | 3,4513 | **−0,0011** |

**PUTUSAN: G2 BELUM LULUS** (ambang +0,05). Kontribusi Tingkat B **+0,008 bit pada 30k lalu
memburuk menjadi −0,016 bit pada 100k** — persis pola yang dokumen peringatkan ("tidak boleh
mengecil saat data bertambah"). Sesuai §7 dokumen: Tingkat B **dikeluarkan dari jalur produksi**,
disimpan sebagai modul riset, hasil negatif ditulis.

⚠️ **Kebocoran yang ditemukan lebih dulu** (versi pertama: +0,143 terlatih, **+0,260 ACAK**):
`nn_st[i]` dihitung dari state yang sudah melihat byte i padahal Tier A memakainya untuk
memprediksi byte i (**off-by-one**), dan pohon biner memakai prefiks byte yang sudah dilihat.
**Pelajaran: model ACAK adalah kontrol WAJIB untuk setiap jalur integrasi.** Ini kebocoran
protokol **ketiga** di proyek ini (setelah `frozen_first` dan probe CE-latih).

**Catatan jujur**: diukur pada 30k–100k byte, bukan 1M–10M yang dokumen minta. Arahnya jelas
(mengecil), skalanya belum penuh. Total suite **88 tes lolos**.

### 19.7 M6 — G4 kurva skala Tingkat A: **LULUS** (2026-10-04)
Skrip baru `scripts/scale_curve.py`. **JENDELA UJI TETAP**: selalu 8000 byte terakhir korpus,
dipakai untuk SEMUA ukuran latih (latih dipangkas agar tak menyentuh uji).

| ntrain (WikiText) | BPB uji | Δ |
|---|---|---|
| 100k | 2,6678 | — |
| 200k | 2,3755 | −0,2923 |
| 400k | 2,2046 | −0,1709 |
| 800k | 2,1263 | −0,0783 |
| 1,10M | **2,0630** | −0,0633 |

**Monoton turun, TANPA plateau** (konfigurasi produksi: `table_bits=25`, `wsel_bits=12`).
**G4 LULUS** pada rentang yang korpus lokal izinkan (maks 1,12 MB — belum 10M/100M).

⚠️ **JEBAKAN PROTOKOL KEEMPAT — jendela uji berpindah.** Versi pertama memakai
`raw[n:n+ntest]`, sehingga tiap baris kurva dinilai di **jendela berbeda** → goyangan palsu
±0,09 bit (2,1963 → 2,2902 → 2,2187, "tidak monoton"). Setelah jendela dibekukan → kurva
mulus monoton. **Angka dari jendela uji berbeda TIDAK BISA dibandingkan.**

⚠️ **`table_bits` adalah segalanya**: pada 200k/8k, `table_bits=22` memberi **2,52**,
`table_bits=25` memberi **1,83**, `table_bits=26` **1,78**. Default `scale_curve.py` sempat 22
dan memunculkan "regresi" palsu 2,52 vs angka historis 1,78 — bukan regresi, hanya salah knob.

### 19.8 M6 — shard multi-inti: KECEPATAN naik, KUALITAS runtuh (2026-10-04)
Skrip baru `scripts/shard_stat_core.py`. N inti, masing-masing satu `StatCore`; jendela uji
sama; probabilitas per-bit dirata-ratakan, lalu BPB. Dua mode.

| N inti | split BPB | full BPB | full agregat B/s | speedup |
|---|---|---|---|---|
| 1 | 2,4437 | 2,4437 | 30.974 | 1,00× |
| 2 | 6,1432 | 5,9888 | 44.075 | 1,42× |
| 4 | 12,1593 | 11,5492 | 62.036 | 2,00× |
| 8 | 21,3100 | 19,3525 | **143.172** | **4,62×** |

**PUTUSAN jujur**: shard = pengungkit **KECEPATAN penilaian** (sampai **4,62× pada 8 inti**,
≈143k B/s agregat), **BUKAN kualitas** — BPB runtuh di KEDUA mode. Penyebab: inti biner sangat
**jenuh** (16,5% bit di-clamp ke 1e-6 / 1−1e-6) dan **deterministik** (dua model dengan `lr`
berbeda memberi BPB identik 2,5077/2,5076) → tak ada keragaman untuk di-ensembel; merata-ratakan
prediksi yang bertentangan menghukum keduanya. Untuk kualitas, **satu tabel pada semua data
tetap terbaik**. G3 (≥500k agregat) belum tercapai pada 8 inti (143k); butuh lebih banyak inti
atau shard pada penilaian-beku saja.

⚠️ **Jebakan satuan (bukan kebocoran)**: rata-rata CE harus per BYTE (×8), bukan per bit. Versi
pertama melaporkan BPB **0,315** — mencurigakan-tampak-bocor, tetapi ternyata tepat `2,522/8`.
Rasio 8× = lupa faktor byte. **Selalu periksa apakah CE dihitung per bit atau per byte.**

⚠️ **Windows `multiprocessing` memakai start-method `spawn`** → `Pool(initializer=...)` hanya
bekerja bila skrip punya guard `if __name__ == "__main__"` dan fungsi worker di level modul.
Tes cepat via `python -c` gagal (`Can't get attribute ... on <module '__main__'>`); skrip
sesungguhnya aman.

Total suite **88 tes lolos** (tidak berubah; kerja M6 hanya skrip baru).


### 19.9 Tingkat A — latih + generasi korpus Indonesia 1M (2026-10-04)
Skrip `scripts/train_gen_id.py`. Konfigurasi produksi (`table_bits=25`, `wsel_bits=12`, APM/match/
word on, sparse OFF). Latih pada **seluruh 957.490 byte** korpus `data/indonesian_1m_pure.txt`.

| metrik | nilai |
|---|---|
| BPB tahan (uji tetap 8.000 B) | **1,8046** |
| waktu latih | 12,6 s |
| throughput | ~80.000 B/s |
| memori tabel | 2²⁵ ≈ 268 MB |

Menang atas TMT2+mixer simbol (2,4809, **+0,155**), **bz2 -9 (2,3700, +0,044)**, n-gram JM-5 (2,5400).

**Generasi**: morfologi & sintaks lokal Indonesia baik (imbuhan `-nya`/`-kan`/`-an`/`me-`/`ber-`;
partikel `dan`/`yang`/`di`/`pada`/`dari`/`untuk`). Koherensi topik jarak-jauh lemah — batas wajar
n-gram byte orde ≤24. Contoh (temp 0,75): *"Indonesia adalah negara kepulauan yang diantaranya adalah
bahasa. ... Sementara, dan mengandung dari terbatas pada masa yang bersama ..."*
Laporan: `results/gen_id_tierA_report.html` · teks: `results/gen_id_tierA.txt`.

⚠️ **BUG GENERASI DIPERBAIKI (kritis).** `_sample_byte` dulu memakai `np.prod(byte_probs(b))` —
mengalikan **p(bit=1) untuk SEMUA bit** seolah tiap bit bernilai 1. Byte biner (bit-1 posisi tinggi,
>127) jadi menang → generasi keluar **byte sampah non-teks**. Perbaikan: kalikan menurut **nilai bit
byte itu** (`p` bila bit 1, `1−p` bila bit 0). Ditemukan lewat uji asap cepat sebelum latih panjang.
Ditambah 2 tes regresi: `test_sample_byte_matches_tree_distribution`,
`test_generate_produces_text_not_binary`. **Total suite 90 lolos.**

---

## 20. Korpus Indonesia 190 MB, kurva-U, dan dua cacat kode (2026-10-05/06)

Ringkas: pada korpus besar muncul **kurva-U** (BPB-uji turun lalu naik). Investigasi bertahap
membuang 7 hipotesis, menemukan **2 cacat kode nyata** dan **1 penyebab dominan** (tabel jenuh).
Lalu **Tingkat B diuji ulang** pada skala penuh 1 jt byte — tetap gagal G2.

### 20.1 Ekstraksi korpus (2026-10-04)
`data/IndonesiNewData.parquet` (143 MB, 838.221 baris, Spark 4.1.1; kolom `text,doc_id,language,
quality_score`) → `scripts/extract_parquet.py` → **`data/indonesian_big.bin` 190.330.359 byte**
dari **670.424 dokumen** (filter `language=='id'`, panjang 80–20.000 karakter; dibuang 2.846
bahasa lain + 164.951 panjang, 2,6 s). Sanity: 206 byte unik, mode-frac 0,148, H0 4,5709 bit.
Catatan data: `quality_score` nyaris konstan (p5 0,651 / p95 0,905) → tak berguna sebagai filter;
panjang dokumen **rata-rata 239 karakter** (potongan pendek, bukan artikel utuh); **`doc_id` tidak
monoton** → dokumen sudah tercampur di sumber.

### 20.2 Pelaporan: CE kumulatif menyesatkan (diperbaiki)
Gejala pertama: "CE-latih naik 2,0710 → 2,1130, makin lambat". **Bukan regresi.** CE **kumulatif**
(rata-rata dari byte ke-0) hampir tak bergerak saat latih panjang, dan lonjakan segmen terakhir
(data sukar) tampak seperti model mundur. Perbaikan: `scripts/train_big_id.py` melaporkan
**CE-seg berjendela** (reset tiap laporan) + **CE-cum** + **BPB-uji berkala** pada jendela tetap.
Setelah itu terlihat **monoton turun** di awal (CE-seg 2,0749 → 2,0663; BPB-uji 2,2280 → 2,2226
pada 10→20 jt byte). Kecepatan "turun 87k → 68k B/s" juga bukan efek ukuran data: 87k = **fase
hangat** (tabel belum terisi, cache hangat); steady-state **~68–84k B/s**.

### 20.3 Kurva-U (gejala)
Dengan jendela uji **tetap** (bukan bergeser), BPB-uji **memburuk** seiring latihan pada korpus
besar: tb=25 **2,1191 (3 jt) → 2,1028 (9 jt) → 2,2744 (30 jt)**; tb=27 1,9379 (12 jt) → 2,2857
(30 jt). Semua ukuran tabel **konvergen ~2,28** pada 30 jt byte — tabel lebih besar hanya
**menunda** titik balik (tb=25 → 9 jt; tb=27 → 12 jt), tidak menghilangkannya.

### 20.4 Dua cacat kode NYATA (ditemukan + diperbaiki)
| # | cacat | bukti | perbaikan |
|---|---|---|---|
| 1 | **checksum match = bit indeks.** `mck = mkey & 255`, padahal `msl = mkey & mmask` juga memakai bit bawah → `mchk[msl] == mck` **selalu benar** untuk slot terisi → model match tak pernah memverifikasi konteks. | **100% dari 299.994 sampel** "cocok" | `mck = (mkey >> 40) & mck_mask` (bit yang tak dipakai indeks) |
| 2 | **Indeks konteks `c` tidak ikut di-hash.** `key = _mix64(hctx[c] ^ c0*GOLDEN)`. Dua konteks berbeda dengan byte sama memetakan ke slot yang sama → saling menimpa. | kapasitas efektif berkurang | `key = _mix64(hctx[c] ^ c0*GOLDEN ^ (c+1)*CCTX)` |

Keduanya dijaga flag `match_fix` / `ctx_fix` agar bisa dibandingkan terkontrol. **Efek gabungan
terukur** (tb=22, 5 jt byte): **2,6371 → 2,5755 = −0,0616 bit** (match_fix −0,047 · ctx_fix
−0,015). Akurasi match naik **0,541 → 0,587**; cakupan turun 0,731 → 0,668 (checksum lebih ketat
menolak match palsu — memang tujuannya).

### 20.5 Tujuh hipotesis yang DIBUANG secara empiris
| hipotesis | hasil | putusan |
|---|---|---|
| ukuran tabel sebagai penyebab tunggal | tb 22/25/26/27 semua konvergen ~2,28 | bukan penyebab tunggal |
| `count_cap` (ambang pangkas penghitung) | 1023 vs 4095 → 2,4832 vs 2,4832 | **nol** |
| lebar checksum `cks_bits` | 8/16/24 bit → 2,470495 / 2,470172 / 2,470173 | −0,0003, bukan penyebab |
| `apm_lr` | −0,007 | kecil |
| korelasi slot↔checksum | ≈0 | bukan penyebab |
| **kesulitan data naik** | bz2 -9 per blok 1 MB = **2,5511–2,5784** sepanjang 30 MB | **dibuang** |
| **urutan/drift dokumen** | eksperimen 2 (acak dokumen): ASLI 2,1605→…→**2,1720** · ACAK 2,1532→…→**2,1714** — kurva-U muncul di **KEDUANYA**, bentuk hampir identik (selisih akhir 0,0006) | **dibuang** |

⚠️ **Jebakan metodologi**: `COUNT_CAP` sebagai **konstanta tingkat-modul diabaikan numba** (dibekukan
saat kompilasi) — perubahan saat runtime **tak berefek** (2,483178 identik untuk 1023/4095).
Setelah dijadikan **parameter fungsi**, efeknya tetap ≈0. Juga: perbandingan lintas-skrip sempat
terkonfound (`cks_bits` 24 vs 8; jendela uji 20 kB bising ±0,05 ≈ 1/3 amplitudo kurva-U).

### 20.6 Eksperimen 1 — BUKTI degradasi INTERNAL (`scripts/diag_blocks.py`)
Satu model dilatih berurutan; **tiap blok 1 MB** diukur CE kita vs **bz2 -9 blok itu**.

| blok | CE kita | bz2 blok | rasio | isi tabel |
|---|---|---|---|---|
| 0 (0–1 MB) | 2,2034 | 2,5545 | 0,863 | 63,5% |
| 3–5 | **2,02–2,04** (minimum) | 2,56 | 0,79 | 97–99% |
| 30 | 2,0797 | 2,5657 | 0,811 | 100% |
| 39 | 2,3042 | 2,5627 | 0,899 | 100% |
| 67 | 2,3477 | 2,5792 | 0,910 | 100% |

Kuartal pertama vs terakhir: **CE 2,0526 → 2,3296 (+0,2771)**, **bz2 2,5655 → 2,5682 (+0,0026)**,
**rasio 0,800 → 0,907 (+0,107)**.
→ **Kesulitan data TIDAK naik; degradasi murni INTERNAL model.** Fase jelas: 0–3 mengisi tabel
(CE turun) · 4–16 dataran · 17–29 naik pelan · 30+ naik tajam (tabel 100% penuh → penimpaan).
**Dengan fix match/ctx ON pun kurva-U TETAP ADA** (fix memperkecil amplitudo 0,0251 → 0,0182,
−28%, dan menurunkan BPB 20 jt: 2,2007 → 2,1714, tapi tidak menghapusnya).

### 20.7 `table_bits=26` dipilih (bukti)
Pada 14 jt byte: **tb=26 CE 1,978 — masih TURUN** vs **tb=25 2,048 — sudah berbalik NAIK**.
`count_cap=63` tidak berpengaruh (2,0514 vs 2,0478). Jadi **tabel 2× lebih besar menunda
penimpaan** → dipakai untuk model produksi. (tb=26 = 2²⁶ slot = 536 MB.)

### 20.8 Model 30 jt byte + baseline ADIL (`scripts/train_gen_big.py`)
Latih **30.000.000 B** `data/indonesian_big.bin`, tb=26, cks=24, wsel=12, APM on, sparse off,
fix ON, orders 1..24 → **505 s · ~59k B/s · tabel 100% terisi**.

| metode (jendela uji SAMA: 200 kB setelah prefix latih) | BPB | TMT2 − metode |
|---|---|---|
| **TMT2 Tingkat A (30 jt byte)** | **2,1865** | — |
| lzma -6 (beku) | 2,2680 | **−0,082** |
| bz2 -9 (beku) | 2,4572 | −0,271 |
| zlib -9 (beku) | 3,2888 | −1,102 |

**Menang atas SEMUA kompresor umum**, termasuk lzma (terkuat). Checkpoint
`checkpoints/id_big_30m.npz` (293 MB terkompresi, 67,1 jt slot) — round-trip load terbukti
(`scripts/gen_from_ckpt.py`). Generasi: temp 0,5 lancar & bermorfologi benar; 0,75 beragam;
1,0 rusak. Contoh: *"…Sulawesi Tengah berjalan tentang pihak…"*.

### 20.9 Tingkat B diuji ulang pada skala penuh (G2)
| pengukuran | hasil |
|---|---|
| B mandiri 250k/8k Indonesia | **2,6654** (Tingkat A di korpus sama = 2,3261) → **kalah 0,34 bit** |
| B mandiri + sebaran τ | 2,7300 (lebih buruk) |
| **Integrasi A+B @1.000.000 B** | A saja 3,2159 · A+B terlatih **3,2164 (Δ −0,0005)** · A+B acak 3,2161 (Δ −0,0001, **kontrol bersih**) · A+B kocok 3,2324 (Δ −0,0165) |

**PUTUSAN: G2 BELUM LULUS pada skala penuh** (ambang +0,05). Neural membawa sinyal temporal
(mengocok urutannya merugikan 0,017) **tetapi tidak menambah informasi** di atas tabel Tingkat A.
Sesuai §7: Tingkat B tetap **modul riset**, hasil negatif ditulis.

★ **Temuan: generasi Tingkat B RUNTUH ke tarikan tetap** — apa pun prompt-nya, setelah ~40 byte
keluaran menjadi teks yang SAMA. **Bukan bug train/eval**: jalur latih (`block_forward_cache`) dan
jalur generasi (`blk.forward`/`step`) terbukti menghitung fungsi **identik** (selisih 1,7e−15).

★ **Inisialisasi peluruhan**: dokumen §4.2 minta τ_h tersebar geometris (2…4096), kode memakai
`bl = 0` → λ = 0,5 untuk **semua** head → horizon awal ≈1 langkah. Sebaran diimplementasikan
(`bl_h = ln(τ_h−1)`) dan **diukur**: 2,7300 vs 2,6654 → **sedikit lebih buruk** dan **tidak**
memperbaiki keruntuhan. Karena λ tetap bergantung data, model bisa belajar memisahkan skala
sendiri; default dikembalikan ke **False** (pemenang terukur). + tes regresi
`test_tau_spread_geometric`. **6 tes gamelan lolos.**

### 20.10 MASALAH YANG SEDANG DIHADAPI (ringkas — rinci di `MASALAH_SAAT_INI.md`)
1. **🔴 Kurva-U Tingkat A (kualitas).** Tabel jenuh → penimpaan → BPB-uji naik setelah ~10–30 jt
   byte. Terbukti internal (bz2 datar). Mitigasi sementara: tb lebih besar (26/27) + berhenti di
   titik optimal. **Belum ada solusi struktural** (partisi/4-way set-associative, LRU/penghitung
   umur, tabel tumbuh).
2. **🔴 Tingkat B tidak menambah nilai (G2 gagal @1 jt).** Jalur neural tertutup untuk kualitas.
3. **🟡 Kecepatan Tingkat B ~1–3 kB/s** (numpy float64 murni) → butuh BLAS/GPU bila mau dilanjutkan.
4. **🟡 Generasi runtuh** (Tingkat B) dan koherensi topik jarak-jauh lemah (kedua tingkat).
5. **🟢 Sudah tuntas**: 2 cacat kode, pelaporan CE, `_sample_byte`, jendela uji, APM (menolong
   +0,058 — **koreksi**: klaim lama "APM merugikan" **SALAH**).

**Langkah berikutnya (berurut)**: (a) uji tabel **4-way set-associative / partisi** pada 30 jt byte
— target langsung kurva-U; (b) uji `count_cap` **turun** (60–255) untuk data non-stasioner;
(c) bila lolos, latih **190 MB penuh** + generasi + laporan akhir; (d) opsional: backend

---

## 21. Mixer v3 — akar drift + fix tervalidasi 30M (2026-10-09, revisi 11)

Rincian penuh: `HASIL_E7B_V3.md`. Mentah: `results/e7b_graft_layer.jsonl`,
`results/vmatrix_fast.jsonl`, `results/gate_v3.jsonl`.

**Akar (D1/D2, turunan dari `stat_core.py:246-256`).** Lapis-1 mixer dilatih
galat-akhir global `(bit−pm)` tanpa faktor `w2` dan tanpa gerbang clamp —
bukan gradien loss mana pun. Set yang jenuh (`|d_k|>16`, gradien sejati = 0)
tetap di-update → hanyut linear tanpa gaya pemulih (clamp-ghost). E7b @30jt
membuktikan kooptasi skala: cangkok `w1@3` saja 3,0769 (memburuk), `w2@3`
saja 5,2059 (runtuh); hanya pasangan @3 pulih (2,1456). Prediksi teori
"w1@3 pulih penuh" GUGUR oleh data — fix harus gabungan.

**Kanal runaway = kolom konteks-jarang (temuan baru, terkontrol).** V0 tb25
suite SAMA: sparse-ON 2,4650/max 1843,6 vs sparse-OFF 2,1929/max 17,1.
Ablasi lama "+0,038" skala-kecil; skala-30M 7× (+0,27).

**Fix: `mixer_mode=1` + jadwal (`nhalf=50000`, floor 0,02).** Matriks 1MB×20:
drift ÷4,3 (Δ +0,0985→+0,0235, max 28,4→2,8). Gerbang 30M (suite tetap
tak-bertumpuk): V3 tb25 2,1189 · tb26 2,0317 · sparse-ON 2,1163 · acak 2,1095
(max|w1| ~1, sat ~0) vs V0 2,1929/2,4650/2,2090. Benjolan +0,011 di 18M ada di
SEMUA run (V0 juga), hilang saat diacak → transien data, bukan mixer.

**Biaya jujur:** −2,6% kecepatan (A/B 180,8k→176,1k B/s); −0,008 di ≤1M
(3/3 konsisten, 1 seed). **Default produksi TIDAK dibalik** (butuh multi-seed
+ 190MB). Rekomendasi: <1M legacy · ≥10M V3. Tes: **124 lolos** (+13:
reproduksibilitas, Fase 0, v3, prod-pin).

**Bug reproduksibilitas yang diperbaiki sekalian:** `step()`/`instrument_stream()`
mengabaikan freeze (`_train_fed` tetap 0) → `_eff_rates()` terpusat;
`_train_fed`+`cnt` masuk checkpoint (v3); `lr2` divergen (`from_config`
0,002 vs produksi 0,003) → `stat_lr2` + tes pin.
BLAS/GPU untuk Tingkat B.
