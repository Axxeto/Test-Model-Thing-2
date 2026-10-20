# TMT2 — model byte-stream online (SSM + RTRL diagonal + Tingkat A biner)

> **📖 Dokumentasi lengkap ada di [`DOKUMENTASI_LENGKAP.md`](DOKUMENTASI_LENGKAP.md)** — rujukan
> tunggal untuk arsitektur, semua hasil terukur, gerbang G0–G5, jebakan protokol, dan cara
> menjalankan. README ini hanya pengantar singkat.
> 🔴 **Masalah terbuka saat ini**: [`MASALAH_SAAT_INI.md`](MASALAH_SAAT_INI.md).

**Status ringkas (2026-10-09)**: **124 tes lolos**. **Jalur produksi = Tingkat A**
(`tmt2/stat_core.py`, inti statistik biner) — WikiText **1,78–1,83** BPB, Indonesia **2,1865**
(30 jt byte) / 1,80 (1M) / 2,33 (200k), pada **~130–180k B/s**. Pada 30 jt byte ia **menang atas
semua kompresor umum** (lzma 2,2680 · bz2 2,4572 · zlib 3,2888). 🟢 **Kurva-U TERVALIDASI
perbaikannya (Mixer v3)**: galat-sendiri per set + jadwal (skala 30M datar 2,03–2,12,
max|w1| ~1; dulunya 2,49/max 1843) — rincian `HASIL_E7B_V3.md`. V3 = **default produksi** sejak gerbang §7 lulus (G-small/mid/30M/190M:
`Config.prod()` + `train_gen_big.py` memakai `mixer_mode=1` polos/V2 terpilih).
Kanal runaway = kolom konteks-jarang (sparse ON→OFF: 2,465→2,193 @30M).
Tingkat B (neural paralel-chunk, `tmt2/gamelan/`) **gagal G2** (Δ −0,0005 @1 jt) → modul riset.

Paket `tmt2`: tumpukan lapisan SSM dengan jejak RTRL diagonal (rekurensi), autograd mini untuk
feedforward, learner online, **inti statistik biner Tingkat A** (`stat_core.py`), dan **Tingkat B
neural paralel-chunk** (`gamelan/`).

## Setup (sekali jalan)
Butuh Python >= 3.10. Satu-satunya dependensi runtime adalah numpy.

    python -m venv .venv
    # Windows:  .venv\Scripts\activate
    # Linux/mac: source .venv/bin/activate
    pip install -r requirements-dev.txt      # atau: pip install -e ".[dev]"

Jalankan tes (**90 tes**: autograd, core/RTRL, learner, jalur cepat, memori, kepala, mixer konteks,
Tingkat A/stat_core, Tingkat B/gamelan):

    pytest -q
    # alternatif tanpa pytest (skrip biasa):
    PYTHONPATH=. python tests/test_core.py
    PYTHONPATH=. python tests/test_learner.py

Jalankan eksperimen belajar:

    python scripts/train.py --data data/periodic.txt --n 6000 --d 32   # teks periodik
    python scripts/train.py --n 6000 --d 32                            # korpus lisensi bawaan
    python scripts/smoke_learn.py 1500 64                              # cek pipa gradien (SGD polos)

Jalur cepat (numba @njit, ~3-4x lebih cepat; butuh `numba`):

    pip install numba
    python scripts/train.py --fast --no_stop --n 20000 --d 32
    python scripts/profile_fast.py 32 3000        # profil per fase jalur cepat

## Struktur
    tmt2/autograd.py   autograd numpy float64 (REKONSTRUKSI dari kontrak tes; lihat catatan)
    tmt2/core.py       langkah lapisan + jejak RTRL, tumpukan, kepala byte
    tmt2/learner.py    semua sinyal belajar + dua jalur optimasi + replay/tidur + checkpoint
    tmt2/model.py      Config + TMT2  (PENGGANTI kompatibel; lihat catatan)
    tmt2/fastpath.py   kernel numba @njit (satuan + terpadu) untuk jalur panas
    tmt2/fast_learner.py  FastLearner: learner dengan arena parameter datar + @njit
    tmt2/tokenizer.py  pembungkus BPE (paket `tokenizers`; V dapat dikonfigurasi)
    tmt2/corpus.py     pemuat korpus lintas-platform (data/licenses/, WikiText, TinyStories, ...)
    data/licenses/     GPL-3 + Apache-2.0 (korpus teks nyata bawaan; 46 KB)
    scripts/train.py        latih online + bandingkan dengan n-gram adaptif
    scripts/bigtest.py      latihan besar: byte/BPE, checkpoint+resume, ablasi flag memori
    scripts/ablate.py       ablasi multi-seed (satu varian sekali jalan bisa: --variants no_think)
    scripts/oracle_probe.py probe salin dengan jalur baca/keluaran di-oracle (diagnostik memori)
    scripts/memory_probe.py probe recall memori (copy_task; --perm/--cycle)
    scripts/fair_baseline.py baseline n-gram ADIL (tokenizer dilatih hanya pada prefiks 20%)
    scripts/profile_step.py profil waktu per fase
    scripts/smoke_learn.py  cek pipa gradien (SGD polos pada CE byte)
    scripts/ablation_mixer.py ablasi mixer protokol ADIL (B/B'/C/C-residual/D, 3 metrik;
                             --eval_order frozen_first (default) mencegah kebocoran tabel)
    scripts/audit_leak.py    AUDIT kebocoran tabel mixer (4 uji T1-T4; lihat LAPORAN §16)
    scripts/bptt_control.py  kontrol algoritma belajar: BPTT terpotong vs RTRL diagonal
    scripts/bptt_gradcheck.py validasi gradien kontrol BPTT vs beda-hingga (LULUS 3,4e-08)
    scripts/generate_id.py   uji GENERASI pada teks Indonesia (sampel beberapa suhu)
    scripts/gen_quality.py   metrik generasi objektif (BPB uji-tahan + penilai n-gram luar)
    scripts/gen_quality_v2.py metrik generasi TIDAK SIRKULAR (LCS antar-sampel, distinct-n,
                             kata-valid vs vocab uji, pembanding PPM orde-8)
    scripts/make_gen_report.py laporan HTML generasi (results/gen_id_report.html)
    scripts/probe_state_info.py  probe info-state: berapa info byte-berikutnya ada di state vs
                             readout (jawab "jalur non-memori"; LAPORAN §18.1)
    scripts/probe_epochs.py      probe epoch: apakah hambatan = rezim online (ternyata BUKAN; §18.2)
    scripts/ngram_ceiling.py     lantai n-gram byte Jelinek-Mercer orde 1..8 (§18.3)
    scripts/mixer_variants.py    skema interpolasi mixer vs WB-penuh vs aritmetik (§18.3)
    scripts/kn_ceiling.py        lantai Kneser-Ney byte (TIDAK menolong; §18.3)
    scripts/fair_online_compare.py papan skor ADIL held-out: TMT2+mixer vs n-gram/ctx-mix/bz2 (§18.5)
    scripts/fair_bpe_split.py  baseline BPE token pada split yang sama (tokenizer pada tr saja)
    scripts/probe_readout_capacity.py probe linear vs GLU vs MLP pada state (apakah readout
                             berkapasitas tinggi mengambil info tambahan? ternyata TIDAK)
    tests/             test_autograd, test_core, test_learner, test_fastpath, test_fast_learner,
                       test_stat_core (Tingkat A), test_gamelan (Tingkat B),
                       test_bptt_control (jalankan: PYTHONPATH=. python tests/<nama>.py)

## TMT2 v2 — Tingkat A / B (diperbarui 2026-10-06)
    tmt2/stat_core.py        TINGKAT A+C: inti statistik biner (numba) — JALUR PRODUKSI
    tmt2/gamelan/gong.py     TINGKAT B: rekurensi linear peluruhan bergantung-data (2 bentuk)
    tmt2/gamelan/saron.py    TINGKAT B: konvolusi kausal + state antar-chunk
    tmt2/gamelan/block.py    TINGKAT B: blok penuh (Saron+Gong+gerbang+SwiGLU+RMSNorm)
    tmt2/gamelan/train.py    TINGKAT B: jalur mundur + gradcheck
    scripts/bench_stat_core.py  bench G1 Tingkat A (latih ntrain / uji ntest, BPB + B/s)
    scripts/scale_curve.py      G4: kurva skala (jendela uji TETAP)
    scripts/shard_stat_core.py  shard multi-inti (kecepatan vs kualitas)
    scripts/train_gen_id.py     latih + generasi Indonesia 1M (jalur produksi)
    scripts/extract_parquet.py  parquet IndonesiaNewData -> data/indonesian_big.bin (190 MB)
    scripts/train_gen_big.py    latih 30 jt byte + generasi (tb=26) — BPB 2,1865
    scripts/gen_from_ckpt.py    muat checkpoint .npz lalu generasi
    scripts/diag_blocks.py      investigasi kurva-U: CE per blok 1 MB vs bz2 per blok
    scripts/tierb_train.py      Tingkat B mandiri + generasi
    scripts/tierb_integrate.py  integrasi A+B @1 jt byte (putusan G2, 2 kontrol)
    scripts/m5_integrate.py     integrasi A+B skala kecil (30k/100k)

**Status ringkas (2026-10-06):**
* **Tingkat A = jalur produksi.** Korpus Indonesia 30 jt byte: **BPB 2,1865**, menang atas
  **semua** kompresor umum (lzma 2,2680 · bz2 2,4572 · zlib 3,2888). ~59k B/s, checkpoint bisa dimuat ulang.
* 🔴 **Masalah terbuka**: **kurva-U** — BPB-uji naik setelah tabel jenuh (terbukti internal;
  bz2 per blok datar). `table_bits=26` menundanya. Rinci: **`MASALAH_SAAT_INI.md`**.
* **Tingkat B gagal G2** pada skala penuh 1 jt byte (Δ −0,0005; kontrol acak bersih) → modul riset.
* **Dua cacat kode** diperbaiki (checksum match; indeks konteks di hash) → −0,0616 bit.
* **Koreksi**: APM **menolong** (+0,058); klaim lama "APM merugikan" salah.
* **124 tes lolos** (termasuk reproduksibilitas freeze/ckpt, diagnostik Fase 0, Mixer v3).

Untuk perintah persis dan semua angka, lihat **`DOKUMENTASI_LENGKAP.md`**;
masalah terbuka: **`MASALAH_SAAT_INI.md`**.

## Catatan penting
* `model.py` TIDAK ikut terunggah, jadi saya menyusun pengganti dari kontrak yang dipakai kode lain
  (kontrak tercantum di docstring-nya). Init dan default Config adalah pilihan saya. Jika Anda punya
  `model.py` asli, pakai itu. `Config.seed` dipakai oleh tes dan skrip; tambahkan jika belum ada.
* `autograd.py` adalah rekonstruksi; `tests/test_autograd.py` (gradcheck beda-hingga) ditulis ulang
  dan lolos untuk semua operator.
* Skrip asli membaca `/usr/share/common-licenses/{GPL-3,Apache-2.0}` (khas Debian/Ubuntu) yang tidak
  ada di Windows. `tmt2/corpus.py` mencoba lokasi itu lalu jatuh ke `data/licenses/` yang dibundel
  (atau sumber numpy sebagai cadangan terakhir); `train/ablate/profile_step/smoke_learn` memakai
  pemuat ini. Inti model (`model/core/learner/autograd`) tidak diubah.
* `FastLearner` memakai gradien "segar": parameter dibaca saat BACKWARD, sedangkan `Learner`
  membacanya saat FORWARD (graf menyimpan data). Jika tidak ada pembaruan parameter di antara
  keduanya, hasilnya identik (~4e-15); dengan pembaruan aktif, keduanya tetap sejalan tapi tidak
  bit-identik. Lihat `tests/test_fast_learner.py`.

## Status terverifikasi (sandbox 1 inti)
* RTRL vs BPTT penuh: galat relatif ~5e-16 **untuk 1 lapisan**; tumpukan terpangkas sebagian
  (lihat bagian "Batas ketelitian gradien" di atas).
* test_learner: halting, PCGrad, target span JEPA, eval tanpa belajar, isolasi tidur, checkpoint
  bulat-balik, determinisme, statistik berpikir, batas state — semua lolos.
* **124 tes lolos** (`pytest -q`): autograd, core/RTRL, learner, jalur cepat (fastpath),
  FastLearner (termasuk eksakness terhadap Learner), jalur memori, dan mixer konteks.
* Belajar konteks: pada teks periodik 4.500 byte, bpb seperempat akhir 0,52 / 2,40 / 1,31 (seed 0/1/2),
  di bawah entropi unigram 4,40. Variasi antar seed besar; **ablasi multi-seed wajib** sebelum klaim.
* Pada teks lisensi 46.507 byte: dengan kepala linear + `deep_grad`, model mencapai **3,794 bpb**
  kumulatif / 3,436 sepertiga akhir — mengalahkan unigram dan n-gram 1, **tetapi masih di atas
  n-gram 2 (2,849) dan gzip -9 (2,633)**.
* WikiText penuh (1.121.681 byte, byte): d=32 -> 2,904, d=64 -> 2,862/2,762. Dengan BPE V=4096:
  2,588 bpb, **tetapi baseline token 1-gram yang ADIL = 2,115 -> masih kalah 0,47 bit**.

## Mixer konteks (Pilar 2) — 2026-10-03 — hasil menentukan
> ⚠️ **KOREKSI (revisi 5).** Tabel orde di bawah **TERKONTAMINASI** dan angkanya **DITARIK**.
> Audit memperbaiki akar masalah: evaluasi beku dijalankan SETELAH evaluasi online pada jendela
> uji yang sama, sehingga tabel n-gram menghafal jendela uji → angka beku ~1 bit terlalu rendah.
> Lihat **§16 `LAPORAN_LENGKAP.md`** dan `scripts/audit_leak.py`. Angka bersih di §16.5.

Mixer menambahkan jalur statistik lokal **eksplisit** ke logits:
`logits += Σ_j w_j · logp_j(ctx)`, dengan `logp_j` = distribusi n-gram online orde-`j`
(Witten-Bell) + satu slot match. Karena `logp_j` konstan terhadap parameter model, gradiennya
eksak (`dCE/dw_j = glo·logp_j`; rel maks **2,8e-9**, `tests/test_fast_learner.py::test_mixer_grad_finite_diff`).

Flag: `Config.mixer_k` (>0 mengaktifkan), `mixer_match`, `mixer_match_orders`, `mixer_init`,
`mixer_lr_mult`. Default orde = `"1,2,3,4,5,6,7,8"`, `mixer_lr_mult=0,1` (**di bawah tinjauan ulang**
karena alasan lamanya diambil dari angka beku yang tercemar — lihat §16).

**Orde konteks = knob kualitas TERBESAR** — ⚠️ **tabel lama DITARIK** (kolom beku tercemar).
Efek orde **jujur** (sapuan bersih 60k/8k, 3 seed): BPB **beku 3,49 → 2,14 (1,35 bit, lalu jenuh)**;
BPB **online 3,45 → 2,00 (1,45 bit)**. Angka beku jujur (mixer-murni) ≈ **2,14 (60k) / 2,38–2,49
(200k)** diukur pada jendela eval yang bersih. Kesimpulan kualitatif "orde konteks adalah knob besar"
**justru lebih kuat** dari klaim lama (0,83 bit salah); rincian §16.6.

**Ablasi** (`scripts/ablation_mixer.py`; protokol adil: jendela uji sama, ≥3 seed, lantai dihitung
dari tabel sebenarnya, varian B′ mixer-murni, C-residual di-*warm-start* dari B′; sekarang dengan
`--eval_order frozen_first` sebagai default):

| varian | WikiText 100k/20k (1 seed) | WikiText 500k/30k (3 seed) |
|---|---|---|
| SSM-saja | 3,21 dingin | 3,36 dingin |
| **B′ mixer-murni** (`W_out=0`, hanya `mix_w`) | **2,515** | **2,377** |
| mixer + `E` | 2,511 | 2,377 |
| mixer+SSM (penuh) | 2,554 | 2,475 |
| C-residual (SSM dari titik B′) | 2,518 | 2,434 |
| SSM+prior lama | 3,23 | 3,38 |
| **B′ − C-residual** (positif = SSM menang) | −0,003 (1 seed) | **−0,015** (3 seed) |

> Catatan: kolom 100k = **1 seed** (dulu disandingkan tanpa label dengan kolom 3 seed — dikoreksi),
> kolom 500k = **3 seed**. Baris "B′−C-residual" karena itu tidak boleh dibandingkan langsung
> lintas kolom; angka 3-seed yang mengikat adalah kolom 500k (§12.5c).

**Kesimpulan (KOKOH)**: mixer jauh mengalahkan SSM-saja (~0,7–1,0 bit) — arah ini kokoh.
Kontribusi marginal SSM **kecil dan tidak stabil terhadap skala**: +0,024 bit di 100k
(protokol online, 3/3 seed) tetapi **negatif di 500k — terkonfirmasi 3 seed: −0,0150 online /
−0,0144 warm / −0,0143 cold, 9/9 sel negatif** (§12.5c). Pada rezim ini **SSM tidak memberi
keunggulan stabil**. Detail, kriteria pra-terdaftar: `LAPORAN_LENGKAP.md` §12.

**Kontrol algoritma (truncated BPTT)**: arsitektur yang sama dengan gradien BPTT terpotong
TIDAK mengalahkan RTRL diagonal pada LR terbaik masing-masing (Markov det, 1 lapis: RTRL 0,857
vs BPTT 0,983). Jadi **hambatan BUKAN algoritma belajar, melainkan arsitektur/pemodelan**
(`scripts/bptt_control.py`, `LAPORAN_LENGKAP.md` §13).

## Kepala keluaran & `deep_grad` (perbaikan 2026-09-30)
Dua perubahan yang mengubah hasil secara mendasar. Sejak 2026-10-01 keduanya adalah **default**
`Config`; nilai lama bisa dipulihkan lewat opsi eksplisit.

* `Config.head="linear"` (default) — kepala `logits = W_out @ l2norm(pred) + byte_bias`, menggantikan
  kepala kosinus ber-`logit_scale` tunggal. `l2norm(pred)` WAJIB: tanpa itu `pred` tak dibatasi
  dan latihan divergen (BPB >1000). Menghilangkan mode collapse (prediksi 3 byte -> 38 byte).
  Nilai lama: `head="cosine"`.
* `Config.deep_grad=True` — gradien loss byte dialirkan ke SELURUH tumpukan, bukan hanya layer
  puncak. Tanpa ini, norm gradien loss byte untuk `L0.*`, `L1.*`, dan encoder `E` **tepat nol**
  (desain "JEPA lokal/stop-grad antar lapisan"), sehingga tumpukan 3 lapis efektif = 1 lapis.
  Catatan: karena kepala linear memakai `W_out` sebagai kamus keluaran, `E` **tidak** menerima
  gradien CE sama sekali — `E` belajar dari loss JEPA per-lapisan. Preset `Config.best(...)`
  mengumpulkan semua opsi terbaik (`head=linear`, `deep_grad`, `lr=1e-2`, `warmup=200`,
  `mem_query_x`, `mem_wq_match_k`, `mem_hetero`, `mem_surprise_fixed`, `beta_max=1.0`).

Hasil pada korpus lisensi (46.507 byte, prekuensial, lr=1e-2, warmup=200, `only_byte_loss`):

| Konfigurasi | BPB kumulatif | 1/3 akhir |
|---|---|---|
| kosinus (awal) | 4,970 | 4,765 |
| + kepala linear | 4,924 | 4,408 |
| + `deep_grad` + hanya loss byte | **3,794** | **3,436** |
| unigram / n-gram 1 / n-gram 2 | 4,561 / 3,650 / 2,849 | 4,741 / 3,697 / 2,794 |
| gzip -9 (referensi keras) | 2,633 | — |

Dengan `deep_grad`, uji overfit 1.000 byte x 60 epoch turun ke **1,62 bit** (sebelumnya plateau
~4,5) — kapasitas terbukti cukup. Catatan: dengan `deep_grad`, loss auxiliary (JEPA/VICReg/stop/
state) justru MERUGI (4,925 vs 3,794), jadi pakai `TrainOpts(only_byte_loss=True)`.

## Jalur memori (`mem_query_x` / `mem_wq_match_k`) — perbaikan 2026-10-01
Kueri baca memori semula dibaca dari `ln(h_t)` (campuran berpeluruhan beberapa token). Diuji dengan
`scripts/oracle_probe.py`: dengan jalur baca/keluaran di-oracle, kueri dari `ln(x_t)` turun monoton
(4,50 -> 2,66 bit) sementara `h_t` tetap datar di ~5,0 = seragam. Dua default sekarang:

* `Config.mem_query_x=True` — kueri `q = l2n(ln(x_t) @ Wq)`, dari token mentah.
* `Config.mem_wq_match_k=True` — init `Wq = Wk` (selaraskan ruang baca & tulis). Tanpa ini `Wq`
  acak + `l2norm` -> arah kueri acak -> jalur baca MATI (plateau chicken-and-egg).
  Bukti oracle (L=4, V=32): `wq_init=rand/ortho` -> 5,01 (gagal) vs `match_k` -> 2,55.

**Efek pada teks alami KECIL** (WikiText 300k byte, d=32, 2 seed): memori mati 3,083 / default
3,118 / fix `d_mem=8` 3,0745 / fix `d_mem=32` 3,157. Fix mengubah kontribusi memori dari **negatif
(−0,035 bit)** menjadi **positif kecil (+0,0085 bit)** — konsisten di 2 seed tetapi magnitudonya
~0,01 bit. `d_mem=32` MERUGIKAN di teks alami (hanya menolong di tugas salin). **Kesimpulan:
hambatan TMT2 ada di jalur NON-MEMORI** (`Wc`/GLU/kepala/ekspresivitas SSM).

## Batas ketelitian gradien (penting untuk penilaian)
* RTRL diagonal **eksak hanya untuk 1 lapisan** (~5e-16 vs BPTT). Pada tumpukan, jejak per-lapisan
  tidak menangkap `dh_upper/dθ_lower` secara TEMPORAL, jadi gradien antar-lapisan terpangkas.
  Ini terukur (galat relatif `L0.Wc`: 1 lap ~3e-10; 2 lap ~6,5; 3 lap ~7,0).
* `Config.deep_grad_gate=True` (default) menambahkan suku `dL/dh·(dh/dx)` sehingga kopling
  DALAM-SATU-LANGKAH eksak (`L0.Wc` 3,11e-01 -> 1,09e-10), tetapi kopling tempora masih hilang.
* **Uji kualitas menunjukkan perbaikan gradien ini TIDAK menolong** (3,1075 lama vs 3,1185
  diperbaiki; 0,011 bit lebih buruk, konsisten 2 seed) -> **plateau bukan karena pemotongan
  gradien**. Hipotesis gugur; jangan ulang.

## Kinerja
`Learner` (autograd): ~180 B/s pada d=32 (119 B/s pada d=64). RTRL hanya ~3% waktu; beban utama
overhead ~5.400 panggilan numpy kecil per byte (backward autograd ~48%, konstruksi graf maju
~24%, optimizer ~13%).

`FastLearner` (jalur panas @njit + arena datar): **~3.050 B/s pada d=32** (dari 2.750 sebelum
optimasi optimizer), **~1.530 B/s pada d=64**, ~514 B/s pada d=128 (steadi state, aliran
>=60.000 byte). Isinya:
* `tmt2/fastpath.py` — kernel satuan (rec_fwd/readout_bwd/head_bwd/jepa_bwd/...) DAN kernel
  terpadu `step_fwd` / `flush_rec` / `state_cap_scatter` / **`adamw_step`** (loop eksplisit,
  scratch buffer, gradien langsung ke arena datar). Semua diverifikasi terhadap autograd:
  galat <=5e-16.
* `tmt2/fast_learner.py` — gradien diakumulasi ke vektor datar per-task; PCGrad, clipping,
  dan AdamW sekali atas vektor datar; `p.data` menjadi view ke arena.
* Kunci percepatan: kernel `_mv`/`_vm`/`_dot` memakai **loop eksplisit** — bentuk broadcast
  `(A*v[None,:]).sum(1)` menulis temporer (m,n) penuh tiap panggilan (loop ~7,2x lebih cepat).
  Catatan: `A @ v` **tidak bisa** dipakai di numba tanpa scipy; selalu pakai loop.
* **`TrainOpts.use_njit_adam=True`** (default): seluruh inti `slow_update` (norm+clip+mask+
  AdamW+theta) dijalankan dalam SATU kernel `fastpath.adamw_step`. Terukur **820 -> 152 us per
  siklus = 5,4x**; numerik setara 4,4e-16. Dampak end-to-end HONEST (bench adil, warmup 500,
  best-of-5): **1,09x @ d=32 / 1,29x @ d=64 / 1,32x @ d=128** (kian besar pada d tinggi karena
  kerja optimizer atas vektor datar kian dominan). `False` = jalur numpy lama (acuan/bitwise).
  **`norm_fast=True`**: `sqrt((x*x).sum())` alih-alih `np.linalg.norm` untuk ntot<=65536
  (2,8x lebih cepat pada N kecil; BLAS menang di N besar).
* **Profil per-byte (d=32/2L, 254 us):** hanya **53 us di dalam njit** (2x `step_fwd` 10,5 +
  2x `flush_rec` 13,5 + kepala 18,5); sisanya ~200 us overhead Python. Sisa fase: `_resolve`
  91, `_stack_step` 65, `_consume` 60, `_flush` 50, `slow_update` 31 us/byte.
* **BIAYA DISPATCH numba = fungsi ARITY**: noop 1-arg 0,23 us vs noop 30-arg **2,53 us**.
  `step_fwd`/`flush_rec` punya 42/43 arg -> ~4,2 us Python/panggilan; 8 panggilan ~20 us/byte.
* **KERNEL TERPADU SELESAI** (`fused_fwd`/`fused_bwd`/`fused_promote_mem`, bit-identik vs
  jalur lama; 77 tes lolos). End-to-end nyata (`bench_fused_e2e.py`, korpus license, 8k byte,
  `slow_every=256`): jalur lama 3.298 B/s → **fused 8.054 B/s (2,44x) @ d=32**;
  1.673 → 2.790 B/s @ d=64.
* **Klaim lama "38.540 B/s" DITARIK.** `scripts/proto_fused_est.py` tidak pernah mengukur
  sebyte penuh (tanpa mundur, jejak lapis-2 parsial, tanpa memori); setelah diperbaiki = 39,98
  us/byte. Plafon jujur ≈ **10–12k B/s @ d=32, nl=2** (`proto_lower_bound.py`: maju saja
  20,85 µs/lapisan). **Target 20k B/s TIDAK tercapai** (end-to-end 8.054 B/s) dan bukan soal
  tuning: mundur didominasi jejak RTRL `3·d²` (~13 µs/lapisan, 55% mundur, irreducible tanpa
  BLAS) + loop `mm` 6–10x di bawah BLAS. Butuh (a) `d` lebih kecil, (b) jejak RTRL
  low-rank/subspace, atau (c) BLAS via scipy (**DILARANG**). Fokus dipindah ke kualitas (BPB).
* `TrainOpts.use_stop=False` mematikan loss stop (yang hanya melatih `ws`/`bs`).

## Di mana hambatan kualitas sebenarnya (2026-10-04)

Pertanyaan lama §"jalur non-memori" **sudah dijawab dengan pengukuran**. WikiText, d=32, L=2,
byte, CE tahan:

| sistem | bit/byte |
|---|---|
| SSM saja (tanpa mixer) | 2,92 |
| **mixer saja** (`out_zero_init`) | **1,843** |
| mixer + SSM | 1,844 |
| lantai byte 5-gram (JM, out-of-sample) | 1,925 |
| BPE unigram adil | 2,115 |

**Kesimpulan:**
* **SSM menyumbang ≈ 0 di atas mixer.** Kontribusinya hanya terasa saat mixer berorde rendah
  (orde-1: +0,066 bit; orde-2: +0,009; orde-8: 0,000) → **SSM ≈ model konteks order-2**.
* **Probe info-state** (CE **uji**, lihat catatan koreksi di `STATUS.md` §17.1): rekurensi
  menambah **+0,295 bit** di atas byte sekarang dan jalur baca menambah **+0,334 bit** di atas
  state mentah — keduanya bekerja. **Kapasitas readout bukan penghambat**: GLU lebih buruk,
  MLP ≈ linear. Yang membatasi = **kandungan state** (≈ n-gram orde-2,5), bukan `Wc`/GLU/kepala.
* **Bukan** rezim online (multi-epoch hanya +0,11 bit), **bukan** smoothing n-gram (KN 1,947
  vs JM 1,925), **bukan** gradien (sudah dibuktikan §P0b).
* **Kualitas TMT2 = kualitas mixer**, dan itu **menang** atas lantai byte 5-gram (1,925) dan
  BPE unigram (2,115). Yang kalah baseline adalah **jalur SSM-saja**, bukan sistem lengkap.
* **Perbaikan mendarat**: `Config.best()` = mixer orde 1..12 + `mixer_match` → 1,8420 → 1,8264
  bit (−0,016, **3/3 seed**). Skrip: `scripts/probe_state_info.py`, `probe_epochs.py`,
  `ngram_ceiling.py`, `mixer_variants.py`, `kn_ceiling.py`.

**Untuk naik lebih jauh**: rekurensi diagonal SSM secara struktural = filter-bank low-pass
(transisi diagonal, tanpa pencampuran antar-saluran) → tak bisa merepresentasikan fitur lokal
tajam orde tinggi. Pilihan: mixer lebih kuat (~0,02-0,03 bit), interpolasi aritmetik
mixer⊕SSM (offline 1,809; ~0,03-0,08 bit), atau rekurensi dengan pencampuran antar-saluran
(memecah trik RTRL diagonal — perubahan besar).

### Jalan ke produksi — 5 gate (rinci: `STATUS.md` §18)
1. **Baseline adil** — **belum lulus (3 seed × 2 korpus)**: pada WikiText TMT2+mixer 1,8264
   mengalahkan semua baseline (n-gram 1,925 · **bz2 -9 kondisional 1,945** · ctx-mix 2,019 ·
   lzma 2,148 · **BPE token 2,213** · zlib 2,563); **pada korpus Indonesia KALAH dari bz2 -9
   (2,370) & lzma -9 (2,404)** — TMT2 2,4809. `scripts/gate1_sweep.py` (paralel, 156 detik).
2. **Nilai SSM** — **MEMBLOKIR**: SSM harus ≥ +0,05 bit di atas mixer (sekarang 0,000).
3. **Skala data** — kurva 1M→10M→100M byte monoton (sekarang korpus maks 1,1 MB).
4. **Benchmark standar** — enwik8/text8/WikiText-103 vs publikasi.
5. **Pengerasan produksi** — throughput (perlu GPU), stabilitas numerik, serving.

Papan skor adil held-out: `scripts/fair_online_compare.py` (latih 200k, uji 8k tak terlihat;
TMT2+mixer **1,8273** vs n-gram JM-5 1,9254 vs ctx-mix-4 2,0186).

## TMT2 v2 — Tingkat A: inti statistik biner (2026-10-04)

Rancangan lengkap ada di `ARCHITECTURE_V2.md` (M0–M6). Yang **sudah jalan**: M0 (alat ukur)
dan **Tingkat A** di `tmt2/stat_core.py`. Tingkat B (neural paralel-chunk) belum ada.

Satu byte = **8 keputusan biner**; tiap konteks memberi `p(bit | prefiks)`, masukan mixer
`stretch(p)`. Biaya per byte `O(bit × input)`, bukan `O(V × orde)` seperti mixer simbol.

```python
from tmt2.model import Config
from tmt2.stat_core import StatCore
m = StatCore.from_config(Config.v2())     # atau StatCore(table_bits=25, ...)
for b in data: m.step(b)                   # online, prekuensial
print(m.bpb(held_out))                     # bit/byte
```

**Hasil terukur** (protokol G1: latih 200k / uji 8k, 1 inti, `scripts/bench_stat_core.py`):

| sistem | WikiText | Indonesia | B/s |
|---|---|---|---|
| **Tingkat A v2** (Mixer-2 + konteks kata, tabel 2²⁵) | **1,7837** | **2,3261** | **~100k** |
| TMT2+mixer simbol (`Config.best`) | 1,8264 | 2,4809 | 2.666 |
| n-gram JM-5 beku | 1,9254 | 2,5400 | — |
| bz2 -9 kondisional | 1,9450 | **2,3700** | — |

**Tingkat A menang atas mixer simbol di kedua korpus** (+0,043 / +0,155) dan **mengalahkan
bz2 -9 di Indonesia** (2,3261 vs 2,3700, **+0,044**).

**M0 — koreksi kecepatan**: angka 8.054 B/s yang dikutip selama ini adalah jalur fused dengan
**mixer MATI**; konfigurasi produksi (`Config.best()`, mixer ON) berjalan **2.666 B/s**.
Tingkat A **~100k B/s (38×)**. Catatan: mesin uji dipakai bersama → B/s berayun ~1,5×,
laporkan best-of-N.

**Status G1 — 2 dari 3**: kecepatan **lulus** (~100k) · Indonesia **lulus** (+0,044 atas bz2) ·
**WikiText +0,043** (butuh +0,05 — meleset 0,007).

Ablasi (WikiText): tabel 2²²→2²⁶ −0,16 · APM −0,034 · Mixer-2 −0,012 · **konteks kata −0,012** ·
pemilih bobot 8→12 −0,023 · match −0,008 · **konteks jarang +0,038 (merugi)**.

**Optimasi kecepatan (50k → 100k, bit-identik)**: hash konteks diangkat keluar loop bit (dihitung
8× padahal identik) · tabel dipadatkan jadi satu int64 (3 akses memori → 1) · simpan `key` saja.

`Config.best()` **tetap** menunjuk konfigurasi simbol lama; v2 hanya lewat `Config.v2()` —
tidak ada regresi diam-diam. Tes paritas: `tests/test_stat_core.py` (6 tes).

## TMT2 v2 — Tingkat B: pelengkap neural paralel-chunk (2026-10-04)

Paket `tmt2/gamelan/` — acuan numpy float64. **Saron** (konvolusi kausal + SiLU), **Gong**
(rekurensi linear peluruhan bergantung-data), **gerbang per head** + SwiGLU + RMSNorm.

**Dualitas yang jadi intinya** — Gong punya DUA bentuk yang harus identik:

```python
U_t = λ_t ⊙ U_{t-1} + (1 − λ_t) k_t v_tᵀ      o_t = q_t U_t
# rekuren   : O(1)/langkah, berurutan      -> inferensi/generasi
# chunkwise : intra-chunk jadi MATMUL      -> latihan (memecah plafon RTRL)
```

```python
from tmt2.gamelan import GamelanBlock
blk = GamelanBlock(d=32, heads=4, conv_k=8, seed=0)
y, st = blk.forward(x, mode="chunk", chunk=64)   # paralel
y2, st2 = blk.forward(x, mode="recurrent")       # sekuensial -> sama
```

**Tes paritas** (`tests/test_gamelan.py`): Gong chunkwise == rekuren **9,15e-14** ·
Saron state antar-chunk == deret penuh **6,66e-16** · blok paralel == decode sekuensial lulus ·
‖U‖ stabil pada τ≈2000 · **gradien vs beda-hingga lulus (rel 2,23e-07)**.

**Kecepatan** (satu blok d=32, T=2048): chunkwise C=64 **14,8 µs/byte (67.628 B/s)** vs
rekuren 57,2 µs/byte → **~4× lebih cepat**.

> **Jebakan**: versi pertama `gong_chunkwise` memakai loop O(C²) → **12× lebih lambat** dari
> rekuren. Dualitas chunkwise hanya berguna bila intra-chunk benar-benar matmul — **paritas
> saja tidak cukup, kecepatan wajib diukur**.

### M5 — integrasi A+B: **G2 BELUM LULUS**

Keluaran Tingkat B diubah jadi `stretch(p(bit|prefiks))` lewat pohon biner, lalu masuk sebagai
**satu masukan mixer** Tingkat A (`StatCore.step(..., nn_st=)`). `scripts/m5_integrate.py`:

| skala | A saja | A + B | Δ |
|---|---|---|---|
| 30k byte, 3 epoch | 3,0716 | 3,0632 | **+0,0084** |
| 100k byte, 5 epoch | 2,9754 | 2,9909 | **−0,0155** |
| **kontrol: B ACAK** | 3,4503 | 3,4513 | **−0,0011** |

**Diuji ULANG pada skala penuh (2026-10-06)** — `scripts/tierb_integrate.py`, **1.000.000 byte**,
jendela uji tetap 50 kB, **2 kontrol wajib**:

| varian | BPB | Δ vs A |
|---|---|---|
| A saja | 3,2159 | — |
| A + B terlatih | 3,2164 | **−0,0005** |
| A + B acak (kontrol) | 3,2161 | −0,0001 → protokol **bersih** |
| A + B kocok (kontrol) | 3,2324 | −0,0165 |

**PUTUSAN: G2 BELUM LULUS** (ambang +0,05), kini terkonfirmasi pada skala yang dokumen minta.
Neural membawa sinyal temporal (kocok merugikan 0,017) tetapi **tidak menambah informasi**.
Sesuai §7 dokumen: Tingkat B **dikeluarkan dari jalur produksi**, modul riset.

**Tingkat B mandiri** (250k/8k): **2,6654** vs Tingkat A **2,3261** → kalah 0,34 bit.
★ **Generasi B runtuh ke tarikan tetap** (prompt-independent setelah ~40 byte) — **bukan** bug
train/eval (fungsi latih vs generasi identik, 1,7e−15). ★ Sebaran τ (`bl=0` → λ=0,5 semua head)
diimplementasikan & diukur: 2,7300 vs 2,6654 → lebih buruk; default = pemenang terukur
(flag `tau_spread`).

> **Kebocoran yang tertangkap kontrol acak**: versi pertama memberi +0,143 (terlatih) dan
> **+0,260 (ACAK)**. Sebabnya off-by-one — `nn_st[i]` dibuat dari state yang sudah melihat byte i,
> padahal dipakai untuk memprediksi byte i. **Model acak adalah kontrol WAJIB untuk setiap jalur
> integrasi**: kalau model acak "menang", yang bocor protokolnya.

**Belum ada**: Bonang (SWA+ALiBi) · L>1 blok bertumpuk · `d`/`H` lebih besar · backend BLAS/GPU
(kini ~1–3 kB/s, numpy float64 murni).

## Mixer v3 — fix kurva-U (2026-10-09, rinci: `HASIL_E7B_V3.md`)

Akar: lapis-1 dilatih galat-akhir global tanpa faktor `w2` + update lanjut di zona
clamp (clamp-ghost) → drift linear + kooptasi skala w1/w2; kanal utama = kolom
konteks-jarang (sparse ON→OFF @30M: 2,465→2,193, max|w1| 1843→17). Fix:
`mixer_mode=1` (galat-sendiri per set) + jadwal per-baris (`--sched 50000`,
lantai 0,02). 30M datar (tb25 2,12 · tb26 2,03, max|w1| ~1; legacy 2,47/2,21),
lulus substansi gerbang §7. Biaya: −2,6% kecepatan, −0,008 di ≤1M.

Rekomendasi skala: <1M bebas (selisih ±0,008) · ≥1M V2-default.
Angka G1 legacy dipin oleh `test_golden_legacy_values`, bukan oleh preset.

## Belum dikerjakan
* 🟡 **Validasi akhir V3** — multi-seed G1-V3 + 190 MB penuh, lalu balik default.
  Kurva-U katastrofiknya sendiri SUDAH hilang (lihat atas).
* 🟡 **Generasi** (`gen_health.py` belum ada) — atraktor pasca-over-adaptasi.
* Kapasitas tabel 4-way/gerbang-sisip (pengungkit kedua, setelah V3 default).
* Uji `count_cap` **turun** (60–255) untuk data non-stasioner (belum pernah diuji).
* Uji `count_cap` **turun** (60–255) untuk data non-stasioner (belum pernah diuji).
* Latih **190 MB penuh** + generasi + laporan akhir (setelah kurva-U ditangani).
* ~~Refactor kernel terpadu~~ **SELESAI** (§Kinerja). Kecepatan bukan lagi prioritas: plafon
  desain numpy-murni ≈10–12k B/s, target 20k tidak realistis tanpa melanggar aturan no-scipy.
* ~~Selidiki jalur NON-MEMORI (`Wc`/GLU/kepala/ekspresivitas SSM)~~ **SELESAI** (§di atas):
  hambatan = kandungan state SSM (≈ orde-2,5), bukan readout. Kualitas = kualitas mixer.
* ~~Gate 2 (nilai SSM)~~ **SELESAI & GAGAL**: Δ −0,0005 @1 jt byte. Keputusan jujur = **pakai
  Tingkat A saja**; Tingkat B hanya dihidupkan bila ada bukti baru.
* Kopling gradien tempora antar-lapisan (butuh O(d³)) — prioritas rendah (terbukti tidak menolong).
