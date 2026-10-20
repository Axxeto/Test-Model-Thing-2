"""Model TMT2: konfigurasi, parameter, dan encoder byte.

CATATAN: berkas ini disusun ulang dari KONTRAK yang dipakai core.py / learner.py / tests
(berkas asli tidak ikut terunggah). Bentuk tensor & atribut sengaja mengikuti pemakaian di
kode; nilai awal (init) dan default Config adalah pilihan saya. Jika Anda punya model.py asli,
pakai itu — selama ia menyediakan atribut yang tercantum di CONTRACT di bawah.

CONTRACT (semua dipakai kode lain):
  Config(...).to_dict(), field: lihat dataclass
  ln_np(x), l2n_np(x)                    -- sama persis dengan layer_norm / l2norm di autograd
  TMT2(cfg):
     .cfg, .layers[l].{Wa,ba,B,K,Wc,Wg1,Wg2,Wo,Af,Bf,Wq,Wm}  (Param)  .{Wk,Wv} (ndarray, beku)
     .params  {nama: Param}  (nama unik = Param.name)      .n_params()
     .E (Param 256xd, kamus byte terikat encoder)  .encode_byte(b) -> Tensor(d)
     .dict_tensor() -> Param E    .target_dict() -> ndarray (256,d) dari encoder target (EMA)
     .tp {nama: ndarray}  .ema_update(m)
     .P[(l,h)] (Param dxd), .H[l] (tuple horizon lapisan l; selalu memuat 1)
     .logit_scale (Param skalar, log-skala), .byte_bias (Param 256)
     .ws (Param d), .bs (Param skalar), .flag (Param d)   -- kepala stop & penanda "berpikir"
"""
from dataclasses import dataclass, asdict
import numpy as np
from .autograd import Param, take


def ln_np(x, eps=1e-5):
    xc = x - x.mean(-1, keepdims=True)
    return xc / np.sqrt((xc * xc).mean(-1, keepdims=True) + eps)


def l2n_np(x, eps=1e-8):
    # Jalur 1-D dipakai ribuan kali per byte (kueri memori, target JEPA, prediksi P).
    # `x @ x` jauh lebih murah daripada `(x*x).sum()` karena hanya 2 operasi numpy
    # (dot + bagi) alih-alih 4 (kali, sum, sqrt, bagi).
    if x.ndim == 1:
        return x / np.sqrt(x @ x + eps)
    return x / np.sqrt((x * x).sum(-1, keepdims=True) + eps)


@dataclass
class Config:
    # --- arsitektur
    d: int = 64
    vocab: int = 256                # ukuran kosakata (256 = byte murni; >256 = token BPE/subword)
    n_layers: int = 3
    expand: int = 2
    d_mem: int = 16
    adapter_rank: int = 4
    horizons: tuple = (1, 4, 16)
    use_innovation: bool = True
    use_memory: bool = True
    beta_max: float = 0.5           # laju tulis memori delta maksimum (dikali kejutan)
    logit_scale0: float = 10.0
    separate_out: bool = False      # True: kamus keluaran W_out terpisah dari encoder E
    head: str = "linear"            # "linear" (terbaik) | "cosine" (LAMA, RUSAK -- lihat catatan)
    #   Kepala kosinus mengikat `byte_bias` ke `pred` yang bergerak -> mode collapse (BPB 4,970).
    #   `head="linear"` (logits = W_out @ l2norm(p) + byte_bias) memperbaikinya ke 3,794.
    #   Default sudah "linear" sejak 2026-10-01; "cosine" hanya untuk reproduksi hasil lama.
    deep_grad: bool = False         # True: gradien loss byte dialirkan ke SEMUA lapisan tumpukan
    deep_grad_gate: bool = True     # True: sertakan suku dL/dh . (dh/dx) pada g_x antar-lapisan.
    #   x_l menggerakkan gerbang a dan u lapisan l, bukan hanya jalur residual y = x + f.
    #   Tanpa suku ini gradien antar-lapisan SALAH (terukur 0,31-0,47 relatif pada L0.Wc).
    #   Dengan suku ini gradien dalam-satu-langkah menjadi EKSAK (~1e-10), tetapi kopling
    #   TEMPORAL lewat rekurensi lapisan atas tetap hilang (jejak RTRL bersifat per-lapisan).
    #   False = perilaku lama (untuk ablasi).
    ngram_learn: bool = False       # True: skala prior n-gram (ng_lam) dipelajari
    ngram_lam0: float = 1.0         # nilai awal ng_lam bila ngram_learn
    out_zero_init: bool = False     # True: W_out diinisialisasi NOL (logit awal 0 dari kepala).
    #   Dipakai untuk varian lantai "mixer murni": bila W_out=0 dan byte_bias=0, logits awal
    #   hanya berasal dari mixer -> mengukur apa yang ditambahkan SSM DI ATAS mixer, bukan di
    #   samping kepala acak. Default False = perilaku lama.
    # ---------------- mixer konteks (Pilar 2) ----------------
    mixer_k: int = 0                # 0 = mati. >0: campuran prior orde 1..mixer_k pada logits.
    #   logits += sum_{o=1..k} w_o * log p_o(ctx) + match model. `w_o` dipelajari sebagai
    #   parameter vektor (`mix_w`, panjang k). Karena log p_o KONSTAN terhadap parameter model,
    #   gradiennya sederhana: dCE/dw_o = glo . logp_o (pola sama seperti ng_lam/g_lam_tot).
    #   Ini memberi prior rank-penuh (n-gram) sebagai basis + residual rank-rendah (SSM).
    #   Alasan: logit SSM terbatas rank <= d+1, sedangkan n-gram statistik rank penuh (B2).
    mixer_match: bool = False       # True: sertakan match model (konteks terpanjang yang cocok)
    #   Terukur 2026-10-04 (WikiText 100k, d=32 L=2, seed 0): match=1 memberi -0,027 bit.
    #   `Config.best()` kini mengaktifkannya (bersama orde 1..12; 3/3 seed menang, -0,016 bit).
    mixer_match_recent: bool = False  # True: match "KEMUNCULAN TERBARU" (bukan hanya deterministik)
    #   `mixer_match` lama mensyaratkan konteks punya TEPAT SATU penerus (`len(d)==1`) -> terlalu
    #   konservatif. Diukur di korpus Indonesia (200k/8k): match lama -0,023 bit.
    #   Indeks "kemunculan terbaru" (K=8/12, cakupan 33,8%, akurasi 67,1%): isolasi vs n-gram
    #   orde-5 memberi -0,101 bit, TETAPI di atas mixer TMT2 hanya **-0,020 bit** (sebagian besar
    #   sudah tertangkap orde tinggi mixer). TIDAK menutup celah ke bz2 -9 (2,370 vs 2,477).
    #   Nonaktif secara default; jalur jarak-jauh masih belum terpecahkan.
    mixer_match_ks: tuple = (8, 12)   # panjang konteks indeks match-terbaru (ambil yang terpanjang)
    mixer_match_conf: float = 0.5     # keyakinan massa pada byte hasil match (sisanya merata)
    # ---------------- v2: TINGKAT A (inti statistik biner, `tmt2/stat_core.py`) ----------------
    #   Rancangan: ARCHITECTURE_V2.md. Diaktifkan lewat `Config.v2()`; `Config.best()` TETAP
    #   menunjuk konfigurasi simbol lama sampai G1 lulus (tanpa regresi diam-diam).
    stat_orders: str = "0,1,2,3,4,5,6,8,12,16,24"   # konteks byte (0 = unigram biner)
    stat_sparse: bool = True          # konteks jarang (M2, BELUM diimplementasikan)
    stat_word: bool = True            # konteks kata (M2, BELUM diimplementasikan)
    stat_hash_bits: int = 25          # 2^bits slot penghitung (2^25 ≈ 168 MB)
    stat_lr: float = 3e-3             # laju mixer lapis-1
    stat_lr2: float = 3e-3            # laju mixer lapis-2 (JANGAN disamakan dgn stat_lr!)
    stat_match_minlen: int = 4        # panjang konteks model match (M2)
    stat_match_bits: int = 20
    stat_match_conf: float = 2.0
    stat_wsel_bits: int = 12          # bit pemilih set bobot mixer (c0 + potongan prev byte)
    stat_apm: bool = True             # APM/SSE dua tahap
    stat_apm_lr: float = 0.02
    stat_table_mode: str = "adaptive"  # "adaptive" | "additive" (additive BELUM ada)
    # --- knob reproduksibilitas + fix kurva-U (§11.2-11.4; diteruskan ke StatCore) ---
    stat_count_cap: int = 1023        # ambang pangkas penghitung (dulu COUNT_CAP global)
    stat_cks_bits: int = 24           # lebar checksum konteks
    stat_match_fix: bool = True       # checksum match dari bit-atas (fix 2026-10-04)
    stat_ctx_fix: bool = True         # indeks konteks ikut di-hash (fix 2026-10-04)
    stat_match_ck_bits: int = 8       # lebar checksum match
    stat_wsel_mode: int = 0           # 0=lama ((prev<<8)|c0) | 1=kelas-karakter (B-02)
    stat_stretch_exact: int = 0       # 1=stretch float eksak untuk APM (B-07)
    stat_w_bound: float = 0.0         # batas |bobot| mixer lapis-1 (0=tanpa batas; E6)
    stat_mixer_freeze_after: int = 0  # bekukan mixer+APM setelah N byte (0=tak pernah; E8)
    # --- Mixer v3 (Fase 1) ---
    stat_mixer_mode: int = 0          # 0=legacy | 1=galat-sendiri + jadwal + L2
    stat_s2_clip: float = 16.0        # batas input lapis-2 (mode 1)
    stat_mix_lr_floor: float = 1.0    # lantai faktor jadwal (1.0 = tanpa jadwal)
    stat_mix_lr_nhalf: int = 0        # kunjungan setengah-waktu (0 = nonaktif)
    stat_mix_l2: float = 0.0          # tarikan L2 ke 1/nctx (0 = nonaktif)
    stat_assoc: int = 0               # 1 = bucket 4-way + eviksi hitungan (#4)
    stat_fast_scan: int = 0           # 1 = pemindaian-tunggal (DIPENSIUNKAN, lebih lambat)
    stat_hit_sel: int = 0             # 1 = selector set-0 per hit-order (A2)
    stat_dropout_permille: int = 0    # 0=mati; mis. 100 = drop 10% konteks orde>=5 saat latih
    stat_match_adapt: int = 0         # 1 = keyakinan match adaptif per run-length (C2-L1)
    # ---------------- v2: TINGKAT C (mixer-2) — placeholder rancangan ----------------
    mix2_sets: int = 4                # jumlah set bobot terpilih-konteks (M3 lanjutan)
    apm_stages: int = 2
    # ---------------- v2: TINGKAT B (pelengkap neural) — BELUM ADA ----------------
    nn_enabled: bool = False          # default mati sampai G2 lulus
    nn_d: int = 32
    nn_layers: int = 2
    nn_conv_k: int = 8
    nn_gong_heads: int = 4
    nn_gong_chunk: int = 64
    nn_window_attn: bool = False
    nn_window: int = 256
    mixer_match_orders: str = "1,2,3,4,5,6,7,8"
    #   Orde konteks yang dipakai mixer. HASIL UKUR (WikiText 200k/20k, d=32, B' murni, seed 1;
    #   results/mixer_orders_probe*.jsonl) -- orde konteks adalah knob PALING BERPENGARUH:
    #     "1,2,3"           -> warm/cold 2,015 bit
    #     "1,2,3,4"         -> 1,606
    #     "1,2,3,4,5,6"     -> 1,232
    #     "1,2,3,4,5,6,7,8" -> 1,182  (OPTIMUM; jenuh di sini)
    #     "2,3,4,5,6,7,8,9,10" -> 1,259  (lebih buruk: tanpa orde-1 & konteks terlalu jarang)
    #   Default lama "2,3,4" sangat kekurangan tenaga. Efek +0,83 bit ini jauh lebih besar
    #   daripada kontribusi SSM mana pun yang pernah terukur (~0,01 bit) -- lihat LAPORAN §14.
    mixer_lr_mult: float = 0.1      # pengali LR untuk mix_w.
    #   mix_w hanya berisi ~8 angka; dengan LR penuh (1,0) Adam menggeser bobotnya bolak-balik
    #   selama evaluasi ONLINE dan merusak hasil. Ukur (WikiText 200k, B' murni, orders 1..8):
    #     mult=1,0 -> warm/cold 1,182 ; mult=0,1 -> 1,138 ; mult=0,01 -> 1,258.
    #   Default 0,1 = LR efektif 10x lebih kecil untuk mixer. Lihat LAPORAN §14.2.
    mixer_init: float = 0.0         # nilai awal tiap w_o (0 = mulai tanpa prior, belajar dari SSM)
    mem_write_grad: bool = False    # True: Wk/Wv jadi Param & penulisan memori dapat dipelajari
    mem_hetero: bool = False        # True: kunci tulis dari konteks SEBELUMNYA (asosiasi hetero)
    mem_surprise_fixed: bool = False  # True: beta = beta_max tetap (lepas dari gerbang kejutan)
    mem_query_x: bool = True        # True: kueri baca q = l2n(ln(x_t) @ Wq), bukan dari ln(h_t).
    #   Default True sejak 2026-10-01. Dengan False, kueri dibaca dari campuran berpeluruhan
    #   `ln(h_t)` sehingga `am = M q` tak berkorelasi dengan penerus -> jalur baca mati.
    mem_wq_match_k: bool = True     # True: init Wq = Wk (selaraskan ruang baca & tulis).
    #   Tanpa ini, Wq acak + l2norm -> arah kueri acak -> am = M q tak berkorelasi dengan
    #   penerus -> jalur baca mati (chicken-and-egg: untuk belajar Wq butuh readout bagus,
    #   untuk belajar readout butuh Wq bagus). Terbukti di scripts/oracle_probe.py:
    #   wq_init=rand/ortho -> 5,01 (gagal); wq_init=match_k -> 2,55 (L=4, lantai 1,25).
    hl_min: float = 1.0             # half-life paling cepat (langkah), log-spaced antar lapisan
    hl_max: float = 512.0           # half-life paling lambat
    seed: int = 0
    # --- berpikir
    think_max: int = 2
    think_cost: float = 0.02        # nat: berpikir hanya bermanfaat jika menurunkan CE > biaya ini
    stop_prior: float = 0.6         # prior geometrik PonderNet
    stop_kl: float = 0.01
    # --- regularizer
    vic_window: int = 32
    vic_gamma: float = 0.5
    vic_cov: float = 0.05
    vic_w: float = 0.1
    state_cap: float = 6.0
    state_w: float = 0.1
    # --- optimasi
    lr: float = 3e-3
    warmup: int = 20
    betas: tuple = (0.9, 0.99)
    wd: float = 0.01
    clip: float = 1.0
    slow_every: int = 16
    fast_lr: float = 0.01
    fast_decay: float = 1e-4
    ema: float = 0.99
    s_decay: float = 0.5
    uncertainty: bool = True
    pcgrad: bool = True
    # --- replay & tidur
    replay: bool = True
    replay_cap: int = 16
    seg_len: int = 32
    sleep_every: int = 512
    sleep_segments: int = 2

    def to_dict(self):
        return asdict(self)

    @classmethod
    def best(cls, **over):
        """Preset konfigurasi TERBAIK hasil pengukuran (lihat STATUS.md).

        Pakai bersama ``TrainOpts(only_byte_loss=True)`` -- auxiliary loss (JEPA/VICReg/stop/
        batas state) terukur MERUGI bila ``deep_grad`` aktif (4,925 vs 3,794 pada lisensi byte),
        dan flag itu ada di TrainOpts, bukan di sini.

            cfg = Config.best(d=32)
            ln = FastLearner(TMT2(cfg), TrainOpts(only_byte_loss=True, think_prob=0.0))
        """
        base = dict(head="linear", deep_grad=True, lr=1e-2, warmup=200,
                    mem_query_x=True, mem_wq_match_k=True, mem_hetero=True,
                    mem_surprise_fixed=True, beta_max=1.0, deep_grad_gate=True,
                    # mixer konteks: knob kualitas TERBESAR yang terukur (§14). Orde 1..8 +
                    # LR mixer kecil = 1,138 bit di WikiText 200k, vs 2,015 untuk orde 1..3.
                    # PEMBARUAN 2026-10-04: orde 1..12 + match model. Terukur WikiText
                    # 200k/8k, d=32 L=2, 3 seed: orde 1..8 = 1,8420 bit ->
                    # orde 1..12 + mixer_match = 1,8264 bit (menang 3/3 seed, -0,0156 bit).
                    mixer_k=1, mixer_match=True,
                    mixer_match_orders="1,2,3,4,5,6,7,8,9,10,11,12", mixer_lr_mult=0.1)
        base.update(over)
        return cls(**base)

    @classmethod
    def v2(cls, **over):
        """Preset TMT2 v2 (ARCHITECTURE_V2.md): TINGKAT A (inti statistik biner) aktif.

        Terukur 2026-10-04, protokol G1 (latih 200k / uji 8k, 1 inti, `bench_stat_core.py`):
            Tingkat A   WikiText **1,7837** · Indonesia **2,3261** bit · **~100-107k B/s**
                        (tabel 2²⁵ = 268 MB; best-of-3)
            (pembanding) TMT2+mixer simbol 1,8264 / 2,4809 @ 2.666 B/s · n-gram JM-5 1,9254 /
                         2,5400 · bz2 -9 kondisional 1,9450 / 2,3700

        Status G1: **kecepatan LULUS** (~100k ≥ 100k) · **Indonesia LULUS** (+0,155 atas mixer
        simbol, **+0,044 atas bz2**) · **WikiText +0,043** atas mixer simbol — **meleset 0,007**
        dari ambang +0,05. Naikkan `stat_hash_bits` ke 26 untuk WikiText 1,7784 (+0,048) dengan
        ~2× memori (537 MB) dan kecepatan ~98k.
        CATATAN: mesin uji dipakai bersama → angka B/s berayun ~1,5×; laporkan best-of-N.
        `stat_sparse=False`: konteks jarang TERUKUR MERUGIKAN (+0,038 bit) — jangan diaktifkan.
        Tingkat B/C lanjutan belum ada.
        """
        d = cls.best().to_dict()
        # B-04(v) (diselaraskan 2026-10-06): match_minlen=6, BUKAN 4. Sweep pada Indonesia
        # (400k latih / 200k uji): ml=4 → 2,1940 · ml=5 → 2,1741 · **ml=6 → 2,1674** ·
        # ml=8 → 2,1676. Jalur vektor `train_gen_big` memang sudah memakai 6 (default ctor);
        # `Config.v2` memakai 4 → DUA jalur berbeda tanpa tercatat (bug B-06/B-04v). Disamakan ke 6.
        d.update(stat_orders="0,1,2,3,4,5,6,8,12,16,24", stat_hash_bits=25, stat_lr=3e-3,
                 stat_match_minlen=6, stat_match_bits=20, stat_match_conf=2.0,
                 stat_wsel_bits=12, stat_apm=True, stat_apm_lr=0.02,
                 stat_sparse=False, stat_word=True, nn_enabled=False)
        d.update(over)
        return cls(**d)

    @classmethod
    def prod(cls, **over):
        """Preset PRODUKSI — sumber kebenaran tunggal (§11.4), sama persis dengan
        `scripts/train_gen_big.py` (model 30 jt byte, BPB 2,1865 legacy).

        Bedanya dari `v2()`: tanpa konteks orde-0, `table_bits=26`, `stat_lr=0.002`.
        Sejak 2026-10-10 (gerbang §7 lulus: G-small/mid/30M/190M): `stat_mixer_mode=1`
        = Mixer v3 galat-sendiri (V2 terpilih: tanpa jadwal). Angka G1 legacy dipin
        oleh `test_golden_legacy_values`, bukan oleh preset ini.
        Tes `test_prod_matches_train_gen_big` memakukan kesetaraan ini: bila skrip
        dan preset divergen lagi, tes gagal.
        """
        d = cls.best().to_dict()
        d.update(stat_orders="1,2,3,4,5,6,8,12,16,24", stat_hash_bits=26,
                 stat_lr=0.002, stat_match_minlen=6,
                 stat_match_conf=2.0, stat_wsel_bits=12, stat_apm=True,
                 stat_apm_lr=0.02, stat_sparse=False, stat_word=True,
                 stat_cks_bits=24, stat_count_cap=1023, stat_match_fix=True,
                 stat_ctx_fix=True, stat_match_ck_bits=8, stat_wsel_mode=0,
                 stat_stretch_exact=0, stat_w_bound=0.0, stat_mixer_freeze_after=0,
                 stat_mixer_mode=1, stat_s2_clip=16.0, stat_mix_lr_floor=1.0,
                 stat_mix_lr_nhalf=0, stat_mix_l2=0.0, stat_assoc=0,
                 stat_fast_scan=0, stat_hit_sel=1, stat_dropout_permille=0,
                 stat_match_adapt=0, stat_match_bits=24,
                 nn_enabled=False)
        d.update(over)
        return cls(**d)

class Layer:
    """Satu lapisan SSM: parameter rekurensi (Wa, ba, B, K), baca (Wc, Wq, Wm), GLU, adaptor."""

    def __init__(self, cfg, l, rng, hl_edges):
        d, e, dm, r = cfg.d, cfg.expand, cfg.d_mem, cfg.adapter_rank
        nm = lambda s: f"L{l}.{s}"
        g = lambda *sh, s=1.0: rng.normal(0.0, s / np.sqrt(sh[0]), sh)
        # "jam": half-life log-spaced di dalam pita lapisan ini -> a awal = 2^(-1/hl)
        hl = np.geomspace(hl_edges[l], hl_edges[l + 1], d)
        a0 = 2.0 ** (-1.0 / hl)
        self.Wa = Param(rng.normal(0.0, 0.02, (d, d)), nm("Wa"))
        self.ba = Param(np.log(a0 / (1.0 - a0)), nm("ba"))
        self.B = Param(g(d, d), nm("B"))
        self.K = Param(np.zeros((d, d)), nm("K"))
        self.Wc = Param(g(d, d), nm("Wc"))
        self.Wq = Param(g(d, dm), nm("Wq"))
        self.Wm = Param(g(dm, d, s=0.5), nm("Wm"))
        self.Wg1 = Param(g(d, e * d), nm("Wg1"))
        self.Wg2 = Param(g(d, e * d), nm("Wg2"))
        self.Wo = Param(g(e * d, d, s=0.5), nm("Wo"))
        self.Af = Param(g(d, r), nm("Af"), fast=True)
        self.Bf = Param(np.zeros((r, d)), nm("Bf"), fast=True)
        # penulisan memori: k, v beku (eksogen, tanpa RTRL). Dengan cfg.mem_write_grad keduanya
        # menjadi Param sehingga penulisan memori dapat dipelajari (gradien terpangkas 1 langkah).
        if cfg.mem_write_grad:
            self.Wk = Param(g(d, dm), nm("Wk"))
            self.Wv = Param(g(d, dm), nm("Wv"))
        else:
            self.Wk = g(d, dm)
            self.Wv = g(d, dm)
        if cfg.mem_wq_match_k:
            # selaraskan kueri baca dengan ruang tulis sejak awal (lihat catatan Config)
            self.Wq.data[:] = getattr(self.Wk, "data", self.Wk)
        self.list = [self.Wa, self.ba, self.B, self.K, self.Wc, self.Wq, self.Wm,
                     self.Wg1, self.Wg2, self.Wo, self.Af, self.Bf]
        if cfg.mem_write_grad:
            self.list += [self.Wk, self.Wv]


class TMT2:
    def __init__(self, cfg=None):
        self.cfg = cfg = cfg or Config()
        rng = np.random.default_rng(cfg.seed)
        d = cfg.d
        edges = np.geomspace(cfg.hl_min, cfg.hl_max, cfg.n_layers + 1)
        self.layers = [Layer(cfg, l, rng, edges) for l in range(cfg.n_layers)]
        self.E = Param(rng.normal(0.0, 1.0, (cfg.vocab, d)), "E")
        # kamus keluaran TERPISAH (opsional): memutus konflik E sebagai encoder + prototipe keluaran.
        # Wajib ada pada kepala linear; opsional (kamus kosinus terpisah) pada kepala kosinus.
        need_out = cfg.separate_out or cfg.head == "linear"
        self.W_out = Param(rng.normal(0.0, 1.0 / np.sqrt(d), (cfg.vocab, d)), "W_out") if need_out else None
        if self.W_out is not None and cfg.out_zero_init:
            self.W_out.data[:] = 0.0                        # lantai mixer murni: kepala nol
        self.byte_bias = Param(np.zeros(cfg.vocab), "byte_bias")
        # skala prior n-gram yang dipelajari (skalar; mat_of False -> tanpa weight decay)
        self.ng_lam = Param(np.array(float(cfg.ngram_lam0)), "ng_lam") if cfg.ngram_learn else None
        # --- mixer konteks (Pilar 2): vektor bobot per orde konteks ---
        self.mixer_orders = tuple(int(x) for x in cfg.mixer_match_orders.split(",")) if cfg.mixer_k > 0 else ()
        self.mix_w = (Param(np.full(len(self.mixer_orders)
                                    + (1 if cfg.mixer_match else 0)
                                    + (1 if cfg.mixer_match_recent else 0),
                                    float(cfg.mixer_init)), "mix_w")
                      if cfg.mixer_k > 0 else None)
        self.logit_scale = Param(np.array(np.log(cfg.logit_scale0)), "logit_scale")
        self.ws = Param(rng.normal(0.0, 0.1, d), "ws")
        self.bs = Param(np.array(0.0), "bs")
        self.flag = Param(rng.normal(0.0, 1.0, d), "flag")
        # horizon per lapisan: lapisan cepat -> pendek, lambat -> semua (selalu ada 1)
        hs, n = tuple(cfg.horizons), cfg.n_layers
        self.H, self.P = {}, {}
        for l in range(n):
            k = len(hs) if n == 1 else 1 + int(round(l * (len(hs) - 1) / (n - 1)))
            self.H[l] = hs[:k]
            for h in self.H[l]:
                self.P[(l, h)] = Param(rng.normal(0.0, 1.0 / np.sqrt(d), (d, d)), f"P{l}_{h}")
        self.params = {}
        for l, L in enumerate(self.layers):
            for p in L.list:
                self.params[p.name] = p
        for p in self.P.values():
            self.params[p.name] = p
        for p in (self.E, self.W_out, self.byte_bias, self.logit_scale, self.ws, self.bs,
                  self.flag, self.ng_lam, self.mix_w):
            if p is not None:
                self.params[p.name] = p
        self.tp = {"E": self.E.data.copy()}          # encoder target (EMA), stop-grad

    # -- encoder
    def encode_byte(self, b):
        return take(self.E, int(b))

    def dict_tensor(self):
        """Kamus keluaran kepala kosinus: W_out bila terpisah, selain itu encoder E."""
        return self.W_out if self.cfg.separate_out else self.E

    def out_dict_name(self):
        if self.cfg.head == "linear" or self.cfg.separate_out:
            return "W_out"
        return "E"

    def target_dict(self):
        return self.tp["E"]

    def ema_update(self, m):
        self.tp["E"] = m * self.tp["E"] + (1.0 - m) * self.E.data

    def n_params(self):
        return int(sum(p.data.size for p in self.params.values()))
