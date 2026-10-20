"""FastLearner: learner online TMT2 dengan jalur panas @njit + arena parameter datar.

Semantik SENGAJA dibuat identik dengan tmt2/learner.py (Learner), tetapi:
  * maju/backward lewat kernel tmt2/fastpath.py (tanpa autograd generik di jalur panas);
  * gradien diakumulasi langsung ke vektor datar per-task (arena), bukan dict per-parameter;
  * PCGrad, clipping, dan AdamW dijalankan sekali atas vektor datar (bukan loop 40 parameter).

Validasi: tests/test_fastpath.py membandingkan lintasan CE FastLearner vs Learner.

  from tmt2.fast_learner import FastLearner, Stream, TrainOpts
"""
import json
from collections import deque
import numpy as np
from .model import Config, TMT2, l2n_np, ln_np
from .core import LayerState, StackState
from .learner import Stream, Entry, TrainOpts, halting, pcgrad, ce_np, W_TASKS, F_TASKS, ALL_TASKS, cfg_depth
from . import fastpath as fp

__all__ = ["FastLearner", "Stream", "TrainOpts"]


def pcgrad_fast(vecs, rng):
    """PCGrad dengan norma precomputed.

    Sama persis dengan learner.pcgrad (urutan permutasi RNG identik, hasil float identik),
    tetapi `orig[kj] @ orig[kj]` dihitung SEKALI per task, bukan sekali per pasangan.
    Pada 47k parameter ini memotong ~1/3 biaya PCGrad.
    """
    keys = list(vecs)
    if len(keys) < 2:
        return {k: v.copy() for k, v in vecs.items()}
    orig = {k: v.copy() for k, v in vecs.items()}
    nrm = {k: float(orig[k] @ orig[k]) for k in keys}
    out = {k: v.copy() for k, v in vecs.items()}
    for i in rng.permutation(len(keys)):
        ki = keys[i]
        for j in rng.permutation(len(keys)):
            kj = keys[j]
            if kj == ki:
                continue
            dot = float(out[ki] @ orig[kj])
            if dot < 0.0:
                out[ki] = out[ki] - dot / (nrm[kj] + 1e-12) * orig[kj]
    return out


class FRec:
    """Catatan satu langkah lapisan untuk jalur panas (padanan core.StepRec)."""
    __slots__ = ("l", "depth", "byte", "x", "h", "y", "M", "SWa", "SB", "SK", "Sba", "useK",
                 "a", "cc", "gh", "below", "prev_w", "beta")

    def __init__(self, l, depth, x, h, y, M, SWa, SB, SK, Sba, useK, a, cc):
        self.l, self.depth = l, depth
        self.byte = None
        self.below = None                    # rec layer l-1 pada langkah tumpukan yang sama
        self.prev_w = None                   # rec langkah SEBELUMNYA (yang menulis M ini)
        self.beta = 0.0                      # laju tulis memori pada langkah ini
        self.x, self.h, self.y, self.M = x, h, y, M
        self.SWa, self.SB, self.SK, self.Sba, self.useK = SWa, SB, SK, Sba, useK
        self.a, self.cc = a, cc              # gerbang & cc: dipakai flush_rec utk d h / d x
        self.gh = None                       # akumulator dL/dh untuk task yang sedang diproses


def cfg_n_layers(cfg):
    return cfg.n_layers


class _BufPool:
    """Kolam buffer lapisan (h, y, S_Wa, S_B, S_K, S_ba); didaur ulang.

    Kernel terpadu menulis ke buffer ini, jadi tidak ada alokasi per operasi. Ukuran
    kolam dihitung dari (horizon maks, kedalaman berpikir, jumlah lapisan) x2 (untuk
    aliran tidur/replay). Jika kolam terlalu kecil, tes eksak vs Learner akan gagal.
    """
    __slots__ = ("b", "i")

    def __init__(self, n, d, dm):
        # urutan HARUS sama dengan pembongkaran di _stack_step:
        # h, S_Wa, S_B, S_K, S_ba, y, M, kp, a, cc   (a & cc dipakai flush_rec utk dh/dx)
        self.b = [(np.zeros(d), np.zeros((d, d)), np.zeros((d, d)), np.zeros((d, d)),
                   np.zeros(d), np.zeros(d), np.zeros((dm, dm)), np.zeros(d),
                   np.zeros(d), np.zeros(d)) for _ in range(n)]
        self.i = 0

    def get(self):
        r = self.b[self.i]
        self.i += 1
        if self.i >= len(self.b):
            self.i = 0
        return r


def snapshot_state(st):
    """Salinan MANDIRI dari StackState: ``h`` dan ``M`` DISALIN, jejak RTRL dinolkan.

    Penting: ``StackState.copy()`` hanya menyalin referensi ``h``/``M`` (lihat invarian di
    ``core.LayerState``: "array tidak pernah dimutasi in-place"). Itu benar untuk ``Learner``
    yang mengalokasi array baru tiap langkah, tetapi SALAH untuk ``FastLearner`` yang menulis
    ke buffer kolam berputar. Snapshot replay harus memakai fungsi ini, bukan ``zero_traces()``.
    """
    n = StackState.__new__(StackState)
    n.layers = []
    for L in st.layers:
        ns = LayerState.__new__(LayerState)
        ns.h, ns.M = L.h.copy(), L.M.copy()
        ns.kp = L.kp.copy()               # konteks hetero harus disalin (buffer kolam!)
        ns.S = {k: np.zeros_like(v) for k, v in L.S.items()}
        n.layers.append(ns)
    n.prev_p1 = list(st.prev_p1)          # l2n_np mengalokasi array baru -> aman dibagi
    for ns in n.layers:
        ns.rec = None                     # jangan rujuk rec aliran hidup dari snapshot replay
    return n


class OnlineNgramN:
    """Witten-Bell interpolasi orde 1..N dengan back-off ke unigram add-1.

    Konteks diberikan pemanggil sebagai tuple token (terlama -> terbaru). ``order=1``
    perilakunya identik dengan implementasi bigram sebelumnya, jadi tes lama tetap berlaku.
    Prior KONSTAN terhadap parameter model: hanya menambah ``lam*log p`` ke logits.
    """
    __slots__ = ("V", "order", "uni", "tabs", "match_ks", "recent")

    def __init__(self, V, order=1, match_ks=()):
        self.V, self.order = V, int(order)
        self.uni = np.ones(V)
        self.tabs = [dict() for _ in range(self.order)]   # tabs[k-1]: konteks panjang k
        self.match_ks = tuple(int(k) for k in match_ks)
        # indeks "kemunculan TERBARU": K-gram -> byte yang mengikutinya pada kemunculan terakhir.
        # Berbeda dari `logprior_match` yang mensyaratkan konteks DETERMINISTIK (len(d)==1).
        # UKURAN JUJUR (korpus Indonesia, 200k/8k): isolasi vs n-gram orde-5 memberi -0,101 bit,
        # TETAPI di atas mixer TMT2 (orde 1..12 + bobot terpelajar) hanya **-0,020 bit** —
        # sebagian besar sudah tertangkap orde tinggi mixer. TIDAK menutup celah ke bz2.
        self.recent = {} if self.match_ks else None

    def update(self, ctx, b):
        self.uni[b] += 1.0
        for k in range(1, self.order + 1):
            if len(ctx) < k:
                break
            c = self.tabs[k - 1].setdefault(ctx[-k:], [{}, 0.0])
            c[0][b] = c[0].get(b, 0.0) + 1.0
            c[1] += 1.0
        if self.recent is not None:
            for k in self.match_ks:
                if len(ctx) >= k:
                    self.recent[ctx[-k:]] = b

    def logprior(self, ctx):
        p = self.uni / self.uni.sum()
        for k in range(1, self.order + 1):
            if len(ctx) < k:
                break
            c = self.tabs[k - 1].get(ctx[-k:])
            if c is None:
                break                                     # konteks lebih panjang pasti juga tak ada
            d, n = c
            lam = n / (n + len(d))                        # Witten-Bell
            p = (1.0 - lam) * p
            for tok, cnt in d.items():
                p[tok] += lam * cnt / n
        return np.log(p)

    def logprior_order(self, ctx, o):
        """log p_o = log P(simbol | konteks o byte TERAKHIR) dengan back-off Witten-Bell ke unigram.

        Berbeda dari ``logprior`` (interpolasi bertingkat menjadi SATU distribusi), di sini
        kita kembalikan distribusi KHUSUS orde ``o`` saja. Betul-betul independen per orde:
        orde ``o`` memakai tabel ``tabs[o-1]`` dan back-off ke unigram.
        Ini yang dipakai mixer: ``logits += sum_o w_o * logprior_order(ctx, o)``.
        """
        p = self.uni / self.uni.sum()
        if o >= 1 and len(ctx) >= o:
            c = self.tabs[o - 1].get(ctx[-o:])
            if c is not None:
                d, n = c
                lam = n / (n + len(d))                    # Witten-Bell
                p = (1.0 - lam) * p
                for tok, cnt in d.items():
                    p[tok] += lam * cnt / n
        return np.log(p)

    def logprior_match(self, ctx, maxlen=8):
        """Match model ringan: konteks terpanjang (<=maxlen) yang punya penerus deterministik.

        Aproksimasi: pakai tabel orde tertinggi yang tersedia; bila konteks orde-``maxlen``
        terlihat tepat SATU penerus berulang, taruh massa besar di sana. Tidak semahal match
        model penuh (tanpa indeks posisi), hanya mengandalkan tabel yang sudah ada.
        """
        p = self.uni / self.uni.sum()
        for o in range(min(maxlen, len(ctx)), 0, -1):
            c = self.tabs[o - 1].get(ctx[-o:]) if o - 1 < len(self.tabs) else None
            if c is None:
                continue
            d, n = c
            if len(d) == 1 and n >= 2:                    # konteks ini selalu diikuti satu simbol
                p = np.full_like(p, 1e-3)
                p[next(iter(d))] += 1.0
                p /= p.sum()
                return np.log(p)
        return None                                        # tak ada match: mixer pakai orde lain

    def logprior_match_recent(self, ctx, conf=0.5):
        """Match "KEMUNCULAN TERBARU": konteks K byte terakhir -> byte penerusnya di kemunculan
        TERAKHIR (bukan hanya bila konteks deterministik). Ambil K terpanjang yang ada.

        Mengembalikan log p (nat) atau None bila tidak ada konteks yang pernah terlihat.
        Massa `conf` ditaruh pada byte hasil match, sisanya merata. Bobot efektifnya tetap
        diatur oleh `mix_w` yang terpelajar.
        """
        if self.recent is None:
            return None
        for k in sorted(self.match_ks, reverse=True):
            if len(ctx) < k:
                continue
            b = self.recent.get(ctx[-k:])
            if b is None:
                continue
            p = np.full(self.V, (1.0 - conf) / (self.V - 1))
            p[b] += conf - (1.0 - conf) / (self.V - 1)
            return np.log(p)
        return None


class FastLearner:
    def __init__(self, model, opts=None):
        self.m, self.cfg = model, model.cfg
        self.o = opts or TrainOpts()
        self.rng = np.random.default_rng(self.o.seed + 12345)
        self.tdict = model.target_dict()
        self.slow = [p for p in model.params.values() if not p.fast]
        self.fast = [p for p in model.params.values() if p.fast]
        # ---- arena parameter datar (layout tetap)
        self.offs, self.sizes = {}, {}
        off = 0
        for p in self.slow:
            self.offs[p.name] = (off, p.data.shape)
            self.sizes[p.name] = p.data.size
            off += p.data.size
        self.ntot = off
        self.gacc = {}                       # task -> gradien datar (ntot,)
        self.touched = set()
        self.adam_m = np.zeros(self.ntot)
        self.adam_v = np.zeros(self.ntot)
        self.ftg = {p.name: {} for p in self.fast}
        self.t_adam = 0
        self.unc = {k: 0.0 for k in W_TASKS}
        self.nacc = 0
        self.lsum = {k: 0.0 for k in ALL_TASKS}
        self.lcnt = {k: 0 for k in ALL_TASKS}
        self.replay, self.live = [], None
        self.n_bytes = 0
        self.stats = {"sleeps": 0, "slow_updates": 0, "depth": np.zeros(cfg_depth(self.cfg), dtype=int)}
        self.ema_loss = {k: None for k in ALL_TASKS}
        self.offmask = 1.0 - np.eye(model.cfg.d)
        self.maxH = max(model.cfg.horizons)
        self._zero_d = np.zeros(model.cfg.d)
        # ---- arena = sumber kebenaran: p.data menjadi VIEW ke self.theta
        self.theta = np.concatenate([p.data.reshape(-1) for p in self.slow])
        for p in self.slow:
            o, shp = self.offs[p.name]
            p.data = self.theta[o:o + p.data.size].reshape(shp)
        self.mat_mask = np.zeros(self.ntot, dtype=bool)
        for p in self.slow:
            if p.data.ndim >= 2:
                o, shp = self.offs[p.name]
                self.mat_mask[o:o + p.data.size] = True
        # ---- offset per lapisan untuk kernel terpadu
        self.loff = {}
        for l in range(cfg_n_layers(self.cfg)):
            d = {}
            for nm in ("Wa", "ba", "B", "K", "Wc", "Wq", "Wm", "Wg1", "Wg2", "Wo",
                       "Wk", "Wv"):
                key = f"L{l}.{nm}"
                if key in self.offs:
                    d[nm] = self.offs[key][0]
            self.loff[l] = d
        self.o_E = self.offs["E"][0]
        self.o_flag = self.offs["flag"][0]
        # ---- scratch + kolam buffer lapisan (menghindari alokasi per operasi)
        d, dm, ed, r = self.cfg.d, self.cfg.d_mem, self.cfg.expand * self.cfg.d, self.cfg.adapter_rank
        self.scr = np.zeros(fp.scratch_size(d, dm, ed, r))
        self.scrb = np.zeros(fp.scratch_size_bwd(d, dm, ed, r))
        self.scrmw = np.zeros(fp.scratch_size_mw(d, dm))
        self.o_gam = np.zeros(dm)                    # dL/d(am) dari pembacaan memori
        self.o_q = np.zeros(dm)                      # kueri baca memori
        self.o_gx = np.zeros(d)                      # dL/dh . (dh/dx): suku gerbang pada g_x
        nbuf = 2 * (self.maxH + 3) * (self.cfg.think_max + 1) * self.cfg.n_layers + 32
        self.pool = _BufPool(nbuf, d, dm)
        self.pool_off = _BufPool(nbuf, d, dm)   # aliran non-online (tidur/eval/generate)
        self._pl = self.pool
        # ---- buffer kerja optimizer (semuanya pra-alokasi, tidak ada alokasi per langkah)
        self._zero_r0 = np.zeros((1, 1))
        self.decay = np.ones(self.ntot)
        self.tmp = np.zeros(self.ntot)
        self.tmp2 = np.zeros(self.ntot)
        self.mask = np.zeros(self.ntot, dtype=bool)
        self.mat_idx = np.nonzero(self.mat_mask)[0]
        # buffer kerja AdamW in-place (tanpa alokasi per langkah)
        self.gbuf = np.zeros(self.ntot)
        self.t1 = np.zeros(self.ntot)
        self.t2 = np.zeros(self.ntot)
        self.mat_of = {p.name: (p.data.ndim >= 2) for p in self.slow}
        # ---- LR khusus parameter keluaran (Eksperimen C) + perekaman gradien
        self.lr_mult = np.ones(self.ntot)
        self.dbg_names = ("logit_scale", "byte_bias", model.out_dict_name())
        self.lrmul_of = {}
        if self.o.out_lr_mult != 1.0:
            for nm in self.dbg_names:
                if nm in self.offs:
                    o = self.offs[nm][0]
                    self.lr_mult[o:o + self.sizes[nm]] = self.o.out_lr_mult
        for p in self.slow:
            self.lrmul_of[p.name] = float(self.lr_mult[self.offs[p.name][0]])
        if self.cfg.mixer_lr_mult != 1.0 and "mix_w" in self.offs:
            o = self.offs["mix_w"][0]
            self.lr_mult[o:o + self.sizes["mix_w"]] = self.cfg.mixer_lr_mult
            self.lrmul_of["mix_w"] = float(self.cfg.mixer_lr_mult)
        self.dbg = []
        self.track_preds = [] if self.o.track_preds else None
        self.ng = (OnlineNgramN(self.cfg.vocab, self.o.ngram_order)
                   if (self.o.ngram_lam > 0 or self.cfg.ngram_learn) else None)
        self.only_mask = None
        if self.o.only_params:                       # isolasi: bekukan semua kecuali daftar ini
            mm = np.zeros(self.ntot)
            for nm in self.o.only_params:
                if nm in self.offs:
                    o = self.offs[nm][0]
                    mm[o:o + self.sizes[nm]] = 1.0
            self.only_mask = mm
            self.only_bool = mm > 0
        # ---- tabel paralel untuk kernel adamw_step (indeks nama: 0..len(slow)-1)
        # idx_of[name] = baris padat; offs_a/sizes_a/mat_a/lrmul_a array paralel.
        self.idx_of = {}
        nslow = len(self.slow)
        self.offs_a = np.zeros(nslow, dtype=np.int64)
        self.sizes_a = np.zeros(nslow, dtype=np.int64)
        self.mat_a = np.zeros(nslow, dtype=np.int64)
        self.lrmul_a = np.ones(nslow)
        for i, p in enumerate(self.slow):
            self.idx_of[p.name] = i
            self.offs_a[i] = self.offs[p.name][0]
            self.sizes_a[i] = self.sizes[p.name]
            self.mat_a[i] = 1 if p.data.ndim >= 2 else 0
            self.lrmul_a[i] = self.lrmul_of[p.name]
        self._tidx_a = np.zeros(nslow, dtype=np.int64)
        self._norm_fast = bool(self.o.norm_fast) and self.ntot <= 65536
        # ---- jalur KERNEL TERPADU (opsional; lihat TrainOpts.use_fused_kernel) --------
        self._fused = False
        if self.o.use_fused_kernel:
            self._build_fused()

    # ------------------------------------------------------------ kernel terpadu
    def _fused_compatible(self):
        cfg = self.cfg
        return (cfg.head == "linear" and cfg.adapter_rank == 0
                and self.m.mix_w is None and self.ng is None
                and cfg.think_max == 0
                and bool(self.o.only_byte_loss))

    def _build_fused(self):
        """Bangun arena state datar + bundle argumen tetap untuk fused_fwd/fused_bwd."""
        if not self._fused_compatible():
            raise ValueError("use_fused_kernel memerlukan head='linear', adapter_rank=0, "
                             "tanpa mixer/ngram, think_max=0, dan only_byte_loss=True")
        cfg, m = self.cfg, self.m
        d, dm, ed, r = cfg.d, cfg.d_mem, cfg.expand * cfg.d, cfg.adapter_rank
        nl = cfg.n_layers
        self.f_d, self.f_dm, self.f_ed, self.f_r, self.f_nl = d, dm, ed, r, nl
        self.f_ssz = fp.state_size_per_layer(d, dm)
        # state datar ganda: f_st = state "saat ini" (baru maju), f_st_prev = state langkah lalu
        # (dipakai backward saat byte berikutnya tiba). Keduanya ditukar tiap langkah.
        self.f_st = np.zeros(nl * self.f_ssz)
        self.f_st_prev = np.zeros(nl * self.f_ssz)
        # loff: (nl,12) offset arena theta [Wa,ba,B,K,Wc,Wq,Wm,Wg1,Wg2,oWo,Wk,Wv]
        self.f_loff = np.zeros((nl, 12), dtype=np.int64)
        names = ("Wa", "ba", "B", "K", "Wc", "Wq", "Wm", "Wg1", "Wg2", "Wo", "Wk", "Wv")
        for l in range(nl):
            for i, nm_ in enumerate(names):
                key = f"L{l}.{nm_}"
                if key in self.offs:
                    self.f_loff[l, i] = self.offs[key][0]
        # bundle: parameter tetap (referensi ke view arena theta)
        par = []
        pnames = ("Wa", "ba", "B", "K", "Wc", "Wq", "Wm", "Wg1", "Wg2", "Wo", "Af", "Bf", "Wk", "Wv")
        for l in range(nl):
            L = m.layers[l]
            row = []
            for nm_ in pnames:
                a = getattr(L, nm_, None)
                row.append(getattr(a, "data", a) if a is not None else np.zeros((d, dm)))
            par.append(tuple(row))
        use_inno = bool(cfg.use_innovation)     # di sini selalu False utk kompatibilitas penuh
        self.f_oWout = self.offs[m.out_dict_name()][0]
        self.f_oBias = self.offs["byte_bias"][0]
        self.f_oPtop = self.offs[f"P{nl - 1}_1"][0]
        self.f_bundle = (
            d, dm, ed, r, cfg.vocab, nl,
            int(cfg.use_memory), int(r > 0), int(cfg.mem_query_x),
            int(cfg.mem_hetero), int(cfg.mem_surprise_fixed), int(use_inno),
            cfg.beta_max, tuple(par), m.W_out.data, m.byte_bias.data,
            m.P[(nl - 1, 1)].data, 0, int(cfg.deep_grad_gate),
            self.f_oWout, self.f_oBias, self.f_oPtop)
        self.f_scr = np.zeros(fp.scratch_size(d, dm, ed, r) + 2 * d)
        self.f_out = np.zeros(d + cfg.vocab + 2 * nl * d + 1)
        self.f_scrb = np.zeros(fp.scratch_size_bwd(d, dm, ed, r) + 8 * d + 2 * cfg.vocab + 3 * d)
        # buffer aktivasi per lapisan (diisi fused_fwd, dibaca fused_bwd -> hindari hitung ulang)
        self.f_aoff = fp.fused_act_size(d, dm, ed, r)
        self.f_act = np.zeros(nl * self.f_aoff)
        self.f_e = -1          # ox0: -1 = buang (flag), >=0 = offset arena, -2 = baris E
        # parameter yang selalu tersentuh gradien byte (prakomputasi; dipakai step_fused)
        tc = {"E", "W_out", "byte_bias", f"P{nl - 1}_1"}
        for l in range(nl):
            p = f"L{l}."
            tc.update((p + "Wc", p + "Wq", p + "Wm", p + "Wg1", p + "Wg2",
                       p + "Wo", p + "Wa", p + "B", p + "ba"))
        self._f_touched = frozenset(tc)
        self._f_has_prev = False
        self._f_act_stale = False      # True setelah slow_update: aktivasi tersimpan basi
        self._f_logits_prev = None
        self._f_byte_prev = 0
        self._f_x_prev = None
        self._fused = True

    def step_fused(self, s, b, train=True):
        """Satu byte lewat KERNEL TERPADU: maju (fused_fwd) + mundur byte sebelumnya (fused_bwd).

        Mengembalikan CE prekuensial byte ini (dari logits byte sebelumnya), persis seperti
        ``step``. Hanya untuk konfigurasi kompatibel (lihat _fused_compatible).
        Tidak menangani mixer/ngram/berpikir/replay -- untuk only_byte_loss murni.
        """
        cfg = self.cfg
        m = self.m
        d = self.f_d
        V = cfg.vocab
        has_prev = self._f_has_prev
        # ---- mundur byte SEBELUMNYA (memakai self.f_st_prev) dengan target byte INI.
        # PENTING (prequensial): gradien CE byte-t dihitung dari logits byte (t-1) dengan
        # target byte t -- bukan dari logits byte t. `glo` dihitung DI DALAM fused_bwd.
        if train and has_prev:
            G = self.gacc.get("byte")
            if G is None:
                G = self.gacc["byte"] = np.zeros(self.ntot)
            ox0 = self.o_E + self._f_byte_prev * d     # lapisan 0, kedalaman 0 -> baris E
            # AKTIVASI TERSIMPAN vs HITUNG ULANG: `fused_fwd` menyimpan hn/c0/cf/z1/z2/gv agar
            # `fused_bwd` tak menghitung ulang pass maju. Aktivasi itu dihitung dgn theta SAAT
            # MAJU. `flush_rec` (referensi) justru MENGHITUNG ULANG dgn theta SAAT MUNDUR. Bila
            # `slow_update` mengubah theta di antara keduanya, aktivasi tersimpan menjadi BASI:
            # galat gradien terukur 1,6e-4 (nl=1) s/d 1,2e-3 (nl=2). Karena itu backward PERTAMA
            # setelah tiap SU wajib menghitung ulang (satu pass maju per SU; ~1/slow_every biaya).
            rc = 1 if self._f_act_stale else 0
            fp.fused_bwd((self.f_bundle, self.f_st_prev, self.f_loff, self._f_x_prev,
                          self._f_logits_prev, int(b), ox0, G, None, self.f_scrb,
                          self.f_act, self.f_aoff, rc))
            self._f_act_stale = False
            # Set parameter yang tersentuh gradien byte TIDAK berubah antar byte utk konfigurasi
            # ini -> prakomputasi sekali, cukup satu union (bukan ~20 update per byte).
            self.touched |= self._f_touched
        # ---- maju byte INI ke self.f_st; CE prekuensial byte ini ikut dihitung di kernel
        x = m.E.data[int(b)].copy()
        lp = self._f_logits_prev if has_prev else self.f_out[d:d + V]   # dummy bila byte pertama
        pb = int(b) if has_prev else -1        # target CE = byte INI (prediksi prekuensial)
        fp.fused_fwd((self.f_bundle, self.f_st, self.f_loff, x, self.f_out, self.f_scr,
                      self.f_act, self.f_aoff, lp, pb))
        ce_pre = float(self.f_out[-1]) if has_prev else None
        # SALIN dulu state maju -> f_st_prev SEBELUM promote: backward byte berikutnya butuh
        # M SEBELUM tulis langkah ini (= Mpre saat ini), karena FRec.M referensi = memori masuk.
        # f_st tetap menjadi state maju yang mengalir antar byte -- JANGAN ditukar.
        self.f_st_prev[:] = self.f_st
        # promosikan Mpost -> Mpre agar maju langkah berikutnya membaca memori SESUDAH tulis ini
        fp.fused_promote_mem(self.f_st, d, self.f_dm, self.f_nl)
        self._f_logits_prev = self.f_out[d:d + V].copy()   # WAJIB salin (f_out dipakai ulang)
        self._f_byte_prev = int(b)
        self._f_x_prev = x
        self._f_has_prev = True
        if train:
            self.n_bytes += 1
            s.seg.append(int(b))
            if ce_pre is not None:
                s.seg_ce += ce_pre
                s.seg_n += 1
                # WAJIB: catat loss byte utk pembaruan ketidakpastian (Kendall). Tanpa ini
                # `unc["byte"]` tak pernah berubah -> bobot `exp(-unc)` tetap 1,0 sementara
                # jalur lama menurunkannya -> `gb` dan theta menyimpang setelah slow_update.
                self._note("byte", ce_pre)
            self.nacc += 1
            if self.nacc >= cfg.slow_every:
                self.slow_update(s)
        return ce_pre

    def reset_fused(self):
        """Nol-kan state arena + riwayat langkah (untuk memulai aliran baru)."""
        self.f_st[:] = 0.0
        self.f_st_prev[:] = 0.0
        self._f_has_prev = False
        self._f_act_stale = False

    def _farr(self, name, task, shape):
        d = self.ftg[name]
        a = d.get(task)
        if a is None:
            a = d[task] = np.zeros(shape)
        return a

    # ------------------------------------------------------------ arena
    def _add(self, task, name, g):
        if name in self.offs:
            off, shp = self.offs[name]
            a = self.gacc.get(task)
            if a is None:
                a = self.gacc[task] = np.zeros(self.ntot)
            a[off:off + g.size] += np.asarray(g, dtype=np.float64).reshape(-1)
            self.touched.add(name)
        else:
            d = self.ftg[name]
            g = np.asarray(g, dtype=np.float64)
            d[task] = g.copy() if task not in d else d[task] + g

    # ------------------------------------------------------------ maju
    def _stack_step(self, x, st, z_star):
        cfg, m = self.cfg, self.m
        zs = None if z_star is None else l2n_np(z_star)
        recs, layers, prev_p1 = [], [], []
        use_ad = cfg.adapter_rank > 0
        xin = x
        for l, L in enumerate(m.layers):
            stl = st.layers[l]
            e, s, use_e = self._zero_d, 0.0, False
            pp = st.prev_p1[l]
            if cfg.use_innovation and zs is not None and pp is not None:
                diff = zs - pp
                e = np.sqrt(cfg.d) * diff
                s = 0.5 * float(diff @ diff)
                use_e = True
            o_h, o_SWa, o_SB, o_SK, o_Sba, o_y, o_M, o_kp, o_a, o_cc = self._pl.get()
            fp.step_fwd(xin, stl.h, stl.S["Wa"], stl.S["B"], stl.S["K"], stl.S["ba"],
                        L.Wa.data, L.ba.data, L.B.data, L.K.data, e, use_e,
                        stl.M, L.Wc.data, L.Wq.data, L.Wm.data, L.Wg1.data, L.Wg2.data,
                        L.Wo.data, L.Af.data, L.Bf.data, cfg.use_memory, use_ad,
                        getattr(L.Wk, "data", L.Wk), getattr(L.Wv, "data", L.Wv),
                        stl.kp, s, cfg.beta_max, cfg.mem_hetero, cfg.mem_surprise_fixed,
                        cfg.mem_query_x,
                        o_h, o_SWa, o_SB, o_SK, o_Sba, o_y, o_M, o_kp, o_a, o_cc, self.scr)
            ns = LayerState.__new__(LayerState)
            ns.h, ns.M, ns.S = o_h, o_M, {"Wa": o_SWa, "B": o_SB, "K": o_SK, "ba": o_Sba}
            r = FRec(l, 0, xin, o_h, o_y, stl.M, o_SWa, o_SB, o_SK, o_Sba, use_e, o_a, o_cc)
            r.below = recs[l - 1] if l > 0 else None     # untuk aliran gradien antar lapisan
            r.prev_w = getattr(stl, "rec", None)         # penulis M ini (langkah sebelumnya)
            r.beta = cfg.beta_max * min(1.0, max(0.0, s)) if cfg.use_memory else 0.0
            ns.rec = r
            ns.kp = o_kp
            recs.append(r)
            layers.append(ns)
            prev_p1.append(l2n_np(o_y @ m.P[(l, 1)].data))
            xin = o_y                                 # stop-grad antar lapisan
        new = StackState.__new__(StackState)
        new.layers, new.prev_p1 = layers, prev_p1
        return recs, new

    # ------------------------------------------------------------ satu byte
    def step(self, s, b, train=True, think=None, fast_ok=True):
        m, cfg = self.m, self.cfg
        self._pl = self.pool if s.online else self.pool_off   # jangan pakai kolam aliran hidup
        if s.online:
            self.live = s
        prev = s.buf.get(s.t)
        ce_pre = None if prev is None else ce_np(prev.logits[prev.k], b)
        if self.track_preds is not None and prev is not None and s.online:
            self.track_preds.append(int(np.argmax(prev.logits[prev.k])))   # hanya aliran utama
        zstar = self.tdict[b]
        x = m.E.data[int(b)].copy()                       # encoder: take(E, b)
        prior, logpriors = None, None
        # PENTING: `s.hist` harus menerima byte ini TEPAT SEKALI per langkah, di SEMUA cabang.
        # Jangan pernah memutuskan "sudah ditambahkan?" dengan membandingkan NILAI
        # (`s.hist[-1] == b`) -- byte berulang ("ll", "  ") akan salah dibuang dan merusak
        # konteks mixer. Pakai flag eksplisit.
        appended = False
        if self.ng is not None:
            K_ng = self.o.ngram_order
            if train and s.online and prev is not None:   # perbarui SETELAH prediksi -> tanpa bocor
                self.ng.update(tuple(s.hist)[-K_ng:], int(b))
            s.hist.append(int(b))
            appended = True
            prior = self.ng.logprior(tuple(s.hist)[-K_ng:])   # mentah; skala di _chain
        if m.mix_w is not None:
            # ---- Pilar 2: satu log-prior PER ORDE (+ match). Tabel n-gram harus mampu
            # menyimpan sampai orde tertinggi yang diminta mixer.
            W = self.cfg.vocab
            if getattr(self, "_mix_ng", None) is None:
                ks = self.cfg.mixer_match_ks if self.cfg.mixer_match_recent else ()
                omax = max(max(m.mixer_orders, default=1), self.o.ngram_order,
                           max(ks, default=0))
                self._mix_ng = OnlineNgramN(W, omax, match_ks=ks)
            ngm = self._mix_ng
            if train and s.online and prev is not None:
                ngm.update(tuple(s.hist)[-ngm.order:], int(b))
            if not appended:
                s.hist.append(int(b))
                appended = True
            ctx = tuple(s.hist)
            logpriors = [ngm.logprior_order(ctx, o) for o in m.mixer_orders]
            if self.cfg.mixer_match:
                lm = ngm.logprior_match(ctx, maxlen=max(m.mixer_orders) if m.mixer_orders else 4)
                logpriors.append(lm if lm is not None else np.zeros(W))
            if self.cfg.mixer_match_recent:
                lr_ = ngm.logprior_match_recent(ctx, conf=self.cfg.mixer_match_conf)
                logpriors.append(lr_ if lr_ is not None else np.zeros(W))
        else:
            self._mix_ng = None
        if not appended:                                  # mixer & ngram mati: tetap catat riwayat
            s.hist.append(int(b))
        # --- mode berpikir (think) -------------------------------------------
        # ATURAN: jumlah langkah berpikir saat EVAL wajib mencerminkan cara model
        # DILATIH. Bila think is None (default), turunkan dari konfigurasi latih:
        #   think_prob > 0  -> latihan kadang berpikir -> eval pakai think_max
        #   think_prob == 0 -> latihan TIDAK PERNAH berpikir -> eval K=0 juga.
        # Gejala bila ini salah: eval memakai K=think_max sementara latihan K=0,
        # sehingga CE eval melonjak (terukur +1,64 bit pada 512 byte acak V=64;
        # +0,21 bit pada lisensi 12k) padahal bobot sudah benar. Ini BUKAN bug
        # kualitas model, hanya ketidakcocokan mode.
        if think is None:
            think = (self.o.think_prob > 0.0)
        K = (cfg.think_max if (think and self.rng.random() < self.o.think_prob) else 0) if train \
            else (cfg.think_max if think else 0)
        e = self._chain(s, x, zstar, K, sample=train, prior=prior, logpriors=logpriors)
        e.byte = b
        for chain in e.chain:                             # byte untuk routing g_x -> encoder E[b]
            for r in chain:
                r.byte = b
        t = s.t + 1
        s.t = t
        s.buf[t] = e
        s.st = e.chain_state
        if train:
            s.zwin.append(x)
            s.zwbyte.append(int(b))
            self.stats["depth"][e.k] += 1
            self._resolve(s, t)
            if not self.o.only_byte_loss:
                self._state_cap(e)
                self._vic(s)
            self._consume(s, fast_ok)
        for j in [j for j in s.buf if j < t - self.maxH - 1]:
            del s.buf[j]
        if not train and t - 1 in s.buf:
            del s.buf[t - 1]
        if s.online and train:
            self.n_bytes += 1
            s.seg.append(b)
            if ce_pre is not None:
                s.seg_ce += ce_pre
                s.seg_n += 1
            if len(s.seg) >= cfg.seg_len:
                if cfg.replay and s.seg_n > 0:
                    self._replay_add(s)
                s.seg, s.seg_ce, s.seg_n = [], 0.0, 0
                s.seg_st0 = snapshot_state(s.st)      # salinan mandiri (bukan referensi kolam)
            if cfg.replay and self.replay and self.n_bytes % cfg.sleep_every == 0:
                self.sleep()
        return ce_pre

    # ------------------------------------------------------------ rantai berpikir
    def _chain(self, s, x, zstar, K, sample, prior=None, logpriors=None):
        m, cfg = self.m, self.cfg
        Ltop = cfg.n_layers - 1
        lin = cfg.head == "linear"
        D = None if lin else l2n_np(m.dict_tensor().data)
        cur, chain, states, lg, sp = s.st, [], [], [], []
        xin, z = x, zstar
        for k in range(K + 1):
            recs, cur = self._stack_step(xin, cur, z)
            for r in recs:
                r.depth = k
            chain.append(recs)
            states.append(cur)
            top = recs[-1]
            p1 = top.y @ m.P[(Ltop, 1)].data
            if lin:
                lg.append(fp.head_fwd_lin(p1, m.W_out.data, m.byte_bias.data))
            else:
                lg.append(fp.head_fwd(p1, D, float(m.logit_scale.data), m.byte_bias.data))
            if prior is not None:
                lam = float(m.ng_lam.data) if m.ng_lam is not None else self.o.ngram_lam
                lg[-1] = lg[-1] + lam * prior             # prior n-gram: konstanta, tanpa gradien
            if m.mix_w is not None and logpriors:
                # ---- Pilar 2: mixer konteks. logits += sum_j w_j * logp_j.
                # logp_j KONSTAN terhadap parameter model -> gradien hanya ke w (lihat _resolve).
                w = m.mix_w.data
                for j in range(len(logpriors)):
                    if logpriors[j] is not None:
                        lg[-1] = lg[-1] + w[j] * logpriors[j]
            sp.append(fp.stop_fwd(top.y, m.ws.data, float(m.bs.data)))
            if k == K:
                break
            if (not sample) and float(sp[-1]) > 0.5:
                break
            xin = ln_np(p1) + m.flag.data
            z = None
        n = len(chain)
        halt = halting(sp)
        kstar = (n - 1) if not sample else (int(self.rng.choice(n, p=halt / halt.sum())) if n > 1 else 0)
        e = Entry()
        e.byte = None
        e.prior = prior                                   # dipakai _resolve untuk gradien ng_lam
        e.logpriors = logpriors                           # dipakai _resolve untuk gradien mix_w
        e.zn = l2n_np(zstar)
        e.chain, e.logits, e.stops, e.halt, e.k = chain, lg, np.array(sp), halt, kstar
        e.chain_state = states[kstar]
        return e

    # ------------------------------------------------------------ loss yang matang
    def _span_target(self, s, i, h):
        return l2n_np(np.mean([s.buf[j].zn for j in range(i + 1, i + h + 1)], axis=0))

    def _note(self, task, v):
        self.lsum[task] += v
        self.lcnt[task] += 1
        old = self.ema_loss[task]
        self.ema_loss[task] = v if old is None else 0.99 * old + 0.01 * v

    def _flush(self, recs, task, deep=False):
        """backward readout + kontraksi RTRL -> scatter langsung ke arena datar task.

        deep=True: sertakan seluruh tumpukan di bawah rec yang diberikan, dan alirkan g_x dari
        layer l ke gh layer l-1 (diproses dari puncak ke bawah). Tanpa ini, loss byte hanya
        melatih layer puncak (lihat catatan deep_grad di model.py).
        """
        cfg, m = self.cfg, self.m
        g = self.gacc.get(task)
        if g is None:
            g = self.gacc[task] = np.zeros(self.ntot)
        use_ad = cfg.adapter_rank > 0
        if deep:
            seen, full = set(), []
            for r in recs:
                cur = r
                while cur is not None and id(cur) not in seen:
                    seen.add(id(cur))
                    full.append(cur)
                    cur = cur.below
            recs = sorted(full, key=lambda r: -r.l)
        for rec in recs:
            if rec.gh is None:
                continue
            gy = rec.gh
            L = m.layers[rec.l]
            lo = self.loff[rec.l]
            ox = -1
            if rec.l == 0:                       # lapisan 0: encoder (kedalaman 0) atau flag (berpikir)
                ox = (self.o_E + rec.byte * cfg.d) if rec.depth == 0 else self.o_flag
            gAf = self._farr(f"L{rec.l}.Af", task, (cfg.d, cfg.adapter_rank)) if use_ad else self._zero_r0
            gBf = self._farr(f"L{rec.l}.Bf", task, (cfg.adapter_rank, cfg.d)) if use_ad else self._zero_r0
            fp.flush_rec(rec.h, rec.x, rec.M, gy, L.Wc.data, L.Wq.data, L.Wm.data,
                         L.Wg1.data, L.Wg2.data, L.Wo.data, L.Af.data, L.Bf.data,
                         rec.SWa, rec.SB, rec.SK, rec.Sba,
                         L.Wa.data, L.B.data, rec.a, rec.cc,
                         rec.useK, cfg.use_memory,
                         use_ad, cfg.mem_query_x, cfg.deep_grad_gate, g, gAf, gBf,
                         lo["Wc"], lo["Wq"], lo["Wm"], lo["Wg1"], lo["Wg2"], lo["Wo"],
                         lo["Wa"], lo["B"], lo["ba"], lo["K"], ox,
                         self.o_gam, self.o_q, self.o_gx, self.scrb)
            if cfg.mem_write_grad and cfg.use_memory and rec.prev_w is not None \
                    and rec.prev_w.beta > 0.0:
                # g_M dari pembacaan rec ini = dL/d(penulisan pada langkah SEBELUMNYA)
                pw = rec.prev_w
                pw_ox = -1
                if pw.l == 0:
                    pw_ox = (self.o_E + pw.byte * cfg.d) if pw.depth == 0 else self.o_flag
                fp.mem_write_bwd(pw.x, pw.M, getattr(L.Wk, "data", L.Wk),
                                 getattr(L.Wv, "data", L.Wv), pw.beta, self.o_gam, self.o_q,
                                 g, lo["Wk"], lo["Wv"], pw_ox, self.scrmw)
                self.touched.update((f"L{pw.l}.Wk", f"L{pw.l}.Wv"))
                if pw_ox >= 0:
                    self.touched.add("E" if pw_ox == self.o_E + pw.byte * cfg.d else "flag")
            pre = f"L{rec.l}."
            self.touched.update((pre + "Wc", pre + "Wq", pre + "Wm", pre + "Wg1",
                                 pre + "Wg2", pre + "Wo", pre + "Wa", pre + "B", pre + "ba"))
            if rec.useK:
                self.touched.add(pre + "K")
            if ox == self.o_E + rec.byte * cfg.d:
                self.touched.add("E")
            elif ox == self.o_flag:
                self.touched.add("flag")
            if deep and rec.l > 0 and rec.below is not None:
                b = rec.below
                # dL/dx_l = dL/dy_l (residual) + dL/dh_l . (dh_l/dx_l) [gerbang & u]
                add = (gy + self.o_gx) if cfg.deep_grad_gate else gy
                b.gh = add if b.gh is None else b.gh + add
            rec.gh = None

    def _add_E(self, task, byte, g):
        """gradien ke baris `byte` dari kamus E (slow param)."""
        off, shp = self.offs["E"]
        a = self.gacc.get(task)
        if a is None:
            a = self.gacc[task] = np.zeros(self.ntot)
        b = int(byte)
        a[off + b * shp[1]: off + (b + 1) * shp[1]] += g
        self.touched.add("E")

    def _resolve(self, s, t):
        m, cfg = self.m, self.cfg
        Lc, Ltop = cfg.n_layers, cfg.n_layers - 1
        lin = cfg.head == "linear"
        # D hanya dipakai cabang kepala kosinus; dengan kepala linear ini pemborosan
        # (normalisasi kamus keluaran penuh) setiap byte.
        D = None if lin else l2n_np(m.dict_tensor().data)
        et = s.buf[t]
        # -- JEPA lokal, kedalaman 0, semua horizon yang matang
        jl, touched = 0.0, []
        if not self.o.only_byte_loss:
          for h in cfg.horizons:
            i = t - h
            if i < 0 or i not in s.buf:
                continue
            tgt = self._span_target(s, i, h)
            ei = s.buf[i]
            for l in range(Lc):
                if h not in m.H[l]:
                    continue
                rec = ei.chain[0][l]
                P = m.P[(l, h)].data
                loss, g_y, gP = fp.jepa_bwd(rec.y @ P, tgt, P, rec.y)
                rec.gh = g_y if rec.gh is None else rec.gh + g_y
                self._add("jepa", f"P{l}_{h}", gP)
                touched.append(rec)
                jl += loss
        # -- JEPA kedalaman berpikir: horizon 1 terhadap z*_t
        if (not self.o.only_byte_loss) and t - 1 in s.buf:
            ei = s.buf[t - 1]
            for k in range(1, len(ei.chain)):
                for l in range(Lc):
                    rec = ei.chain[k][l]
                    P = m.P[(l, 1)].data
                    loss, g_y, gP = fp.jepa_bwd(rec.y @ P, et.zn, P, rec.y)
                    rec.gh = g_y if rec.gh is None else rec.gh + g_y
                    self._add("jepa", f"P{l}_1", gP)
                    touched.append(rec)
                    jl += loss
        if touched:
            self._flush(touched, "jepa")
            self._note("jepa", jl)
        # -- byte + stop (prediksi langkah t-1 vs byte t)
        if t - 1 in s.buf and et.byte is not None:
            ei, b = s.buf[t - 1], et.byte
            n = len(ei.chain)
            ces = np.array([fp.ce_val(ei.logits[k], b) for k in range(n)])
            top_recs, gE_tot, g_ls_tot, g_bb_tot, g_lam_tot = [], 0.0, 0.0, np.zeros(cfg.vocab), 0.0
            g_mixw = (np.zeros(m.mix_w.data.shape) if m.mix_w is not None else None)
            Ptop = m.P[(Ltop, 1)].data
            lin = cfg.head == "linear"
            if lin:                                  # kamus keluaran besar (V=4096) -> scatter langsung
                gflat = self.gacc.get("byte")
                if gflat is None:
                    gflat = self.gacc["byte"] = np.zeros(self.ntot)
                oW = self.offs[m.out_dict_name()][0]
            for k in range(n):
                if ei.halt[k] <= 1e-6:
                    continue
                top = ei.chain[k][-1]
                glo = fp.softmax_grad_target(ei.logits[k], b, float(ei.halt[k]))
                pr = getattr(ei, "prior", None)
                if m.ng_lam is not None and pr is not None:
                    g_lam_tot += float(glo @ pr)          # dCE/dlam = glo . logprior
                if m.mix_w is not None:
                    lps = getattr(ei, "logpriors", None)
                    if lps:
                        for j in range(len(lps)):
                            if lps[j] is not None:
                                g_mixw[j] += float(glo @ lps[j])   # dCE/dw_j = glo . logp_j
                if lin:
                    g_y, gP, g_bb = fp.head_bwd_lin_acc(top.y @ Ptop, m.W_out.data, glo, 1.0,
                                                        Ptop, top.y, gflat, oW)
                    g_bb_tot += g_bb
                else:
                    g_y, gP, gE, g_ls, g_bb = fp.head_bwd(top.y @ Ptop, D, m.dict_tensor().data,
                                                          float(m.logit_scale.data), glo, 1.0, Ptop, top.y)
                    g_ls_tot += g_ls
                    gE_tot = gE_tot + gE
                    g_bb_tot = g_bb_tot + g_bb
                top.gh = g_y if top.gh is None else top.gh + g_y
                self._add("byte", f"P{Ltop}_1", gP)
                top_recs.append(top)
            if top_recs:
                self._flush(top_recs, "byte", deep=cfg.deep_grad)
                if lin:
                    self.touched.add(m.out_dict_name())
                else:
                    self._add("byte", m.out_dict_name(), gE_tot)
                    self._add("byte", "logit_scale", np.array(g_ls_tot))
                self._add("byte", "byte_bias", g_bb_tot)
                if m.ng_lam is not None:
                    self._add("byte", "ng_lam", np.array(g_lam_tot))
                if m.mix_w is not None:
                    self._add("byte", "mix_w", g_mixw)
            self._note("byte", float(np.dot(ei.halt, ces)))
            if self.o.use_stop and (not self.o.only_byte_loss) and n > 1:
                gsp = fp.stop_grads(ei.stops, ces, cfg.think_cost, cfg.stop_kl, cfg.stop_prior)
                gws = np.zeros(cfg.d)
                gbs = 0.0
                for k in range(n):
                    top = ei.chain[k][-1]
                    z = float(top.y @ m.ws.data + m.bs.data)
                    spk = 0.5 * (1.0 + np.tanh(0.5 * z))
                    dz = gsp[k] * spk * (1.0 - spk)
                    gws += dz * top.y
                    gbs += dz
                self._add("stop", "ws", gws)
                self._add("stop", "bs", np.array(gbs))
                self._note("stop", fp.stop_loss(ei.stops, ces, cfg.think_cost, cfg.stop_kl, cfg.stop_prior))

    def _state_cap(self, e):
        cfg = self.cfg
        g = self.gacc.get("state")
        if g is None:
            g = self.gacc["state"] = np.zeros(self.ntot)
        tot = 0.0
        for rec in e.chain[e.k]:
            lo = self.loff[rec.l]
            p = fp.state_cap_scatter(rec.h, rec.SWa, rec.SB, rec.SK, rec.Sba, rec.useK,
                                     cfg.state_cap, cfg.state_w, g,
                                     lo["Wa"], lo["B"], lo["ba"], lo["K"])
            if p > 0.0:
                tot += p
                pre = f"L{rec.l}."
                self.touched.update((pre + "Wa", pre + "B", pre + "ba"))
                if rec.useK:
                    self.touched.add(pre + "K")
        if tot > 0.0:
            self._note("state", tot)

    def _vic(self, s):
        """VICReg atas jendela latent encoder; gradien analitik, langsung ke baris E.

        loss = vic_w * ( mean_j relu(gamma - std_j) + vic_cov * (1/d) * sum_{j!=k} cov[j,k]^2 )
        dengan Zc = Z - mean(Z), std_j = sqrt(mean_i Zc[i,j]^2 + 1e-4), cov = Zc^T Zc/(n-1).
        d lc/dZc = (4/d)/(n-1) * (Zc @ (cov*offmask))
        """
        cfg = self.cfg
        n = len(s.zwin)
        if n < 8 or s.t % self.o.vic_every:
            return
        d = cfg.d
        Z = np.stack(list(s.zwin))
        Zc = Z - Z.mean(axis=0, keepdims=True)
        std = np.sqrt((Zc * Zc).mean(axis=0) + 1e-4)
        lv = float(np.maximum(cfg.vic_gamma - std, 0.0).mean())
        cov = (Zc.T @ Zc) / float(n - 1)
        C = cov * self.offmask
        lc = float((C * C).sum() / d)
        loss = (lv + cfg.vic_cov * lc) * cfg.vic_w
        act = (std < cfg.vic_gamma).astype(np.float64)
        gZc = -(act / d)[None, :] * Zc / (n * std[None, :])          # d lv/dZc
        gZc = gZc + cfg.vic_cov * (4.0 / d) / (n - 1) * (Zc @ C)     # + vic_cov * d lc/dZc
        gZ = cfg.vic_w * gZc
        gZ = gZ - gZ.mean(axis=0, keepdims=True)                     # lewat Zc = Z - mean(Z)
        for i, b in enumerate(s.zwbyte):
            self._add_E("vic", int(b), gZ[i])
        self._note("vic", loss)

    # ------------------------------------------------------------ optimasi
    def _weight(self, task):
        return float(np.exp(-self.unc[task])) if (task in W_TASKS and self.cfg.uncertainty) else 1.0

    def _consume(self, s, fast_ok):
        cfg = self.cfg
        if fast_ok and self.only_mask is None:
            w = self._weight                              # hoist: nilainya sama untuk semua param
            decayi = 1.0 - cfg.fast_decay
            for p in self.fast:
                d = self.ftg[p.name]
                if not d:
                    continue
                if len(d) == 1:                           # kasus umum (only_byte): 1 task
                    (k, v), = d.items()
                    g = v if w(k) == 1.0 else w(k) * v
                else:
                    g = sum(w(k) * v for k, v in d.items())
                # np.linalg.norm punya overhead Python besar untuk array kecil;
                # dot product + sqrt jauh lebih murah dan hasilnya identik.
                gr = g.ravel()
                nn = np.sqrt(gr @ gr)
                # pembaruan IN-PLACE: hindari 2 alokasi per parameter per byte
                p.data *= decayi
                if nn > 1.0:
                    p.data -= (cfg.fast_lr / nn) * g
                else:
                    p.data -= cfg.fast_lr * g
                d.clear()
        else:
            for p in self.fast:
                self.ftg[p.name].clear()
        self.nacc += 1
        if self.nacc >= cfg.slow_every:
            self.slow_update(s)

    def slow_update(self, s=None):
        cfg = self.cfg
        if not self.gacc:
            self.nacc = 0
            return
        K = float(self.nacc)
        wt = [k for k in W_TASKS if k in self.gacc]
        gb, t1, t2 = self.gbuf, self.t1, self.t2
        if len(wt) == 1:                              # jalur cepat: satu task (kasus only_byte)
            np.divide(self.gacc[wt[0]], K, out=gb)
            gb *= self._weight(wt[0])
        else:
            vecs = {k: self._weight(k) * (self.gacc[k] / K) for k in wt}
            if cfg.pcgrad:
                vecs = pcgrad_fast(vecs, self.rng)
            gb[:] = 0.0
            for v in vecs.values():
                gb += v
        for k in F_TASKS:
            if k in self.gacc:
                gb += self.gacc[k] / K
        self.t_adam += 1
        lr = cfg.lr * min(1.0, self.t_adam / max(cfg.warmup, 1))
        b1, b2 = cfg.betas
        # ---- inti optimizer dalam SATU kernel njit: norm+clip+mask+AdamW+theta.
        # Mengganti ~20 dispatch numpy atas 42k-137k float (8-33 us masing-masing) dengan
        # satu lintasan. Terukur 5,4x (820 -> 152 us @ d=32); numerik setara ~1e-15.
        if self.o.use_njit_adam:
            tidx = self._tidx_a
            ni = 0
            for name in self.touched:
                i = self.idx_of.get(name)
                if i is not None:
                    tidx[ni] = i
                    ni += 1
            bc1_inv = 1.0 / max(1e-30, 1.0 - b1 ** self.t_adam)
            bc2_inv = 1.0 / max(1e-30, 1.0 - b2 ** self.t_adam)
            om = self.only_mask if self.only_mask is not None else None
            fp.adamw_step(gb, self.adam_m, self.adam_v, self.theta, self.t1,
                          lr, b1, b2, bc1_inv, bc2_inv, cfg.clip, cfg.wd,
                          self.offs_a, self.sizes_a, self.mat_a, self.lrmul_a,
                          tidx[:ni], om)
        else:
            # norm: sqrt((x*x).sum()) 2x lebih murah dari np.linalg.norm untuk ntot<~50k
            if self._norm_fast:
                gn = float(np.sqrt((gb * gb).sum()))
            else:
                gn = float(np.linalg.norm(gb))
            if gn > cfg.clip:
                gb *= cfg.clip / gn
            if self.only_mask is not None:
                gb *= self.only_mask
            t1, t2 = self.t1, self.t2
            self.adam_m *= b1
            np.multiply(gb, 1.0 - b1, out=t2)
            self.adam_m += t2
            np.multiply(gb, gb, out=t2)
            self.adam_v *= b2
            np.multiply(t2, 1.0 - b2, out=t1)
            self.adam_v += t1
            t1[:] = self.adam_v
            t1 /= (1.0 - b2 ** self.t_adam)
            np.sqrt(t1, out=t1)
            t1 += 1e-8
            t2[:] = self.adam_m
            t2 /= (1.0 - b1 ** self.t_adam)
            np.divide(t2, t1, out=t2)
            dcy = 1.0 - lr * cfg.wd
            for name in self.touched:
                o = self.offs[name][0]
                sz = self.sizes[name]
                v = self.theta[o:o + sz]
                if self.mat_of[name]:
                    v *= dcy
                gs = t2[o:o + sz]
                ml = lr * self.lrmul_of[name]
                if ml == 1.0:
                    v -= gs
                else:
                    v -= ml * gs
        if self.o.debug_out:
            self.dbg.append((self.t_adam, lr, cfg.wd, {
                nm: float(np.linalg.norm(gb[self.offs[nm][0]:self.offs[nm][0] + self.sizes[nm]]))
                for nm in self.dbg_names if nm in self.offs}))
        if cfg.uncertainty:
            for k in W_TASKS:
                if self.lcnt[k]:
                    Lbar = self.lsum[k] / self.lcnt[k]
                    sk = self.unc[k] - self.o.unc_lr * (1.0 - np.exp(-self.unc[k]) * Lbar)
                    self.unc[k] = float(np.clip(sk, -self.o.unc_clip, self.o.unc_clip))
        self.lsum = {k: 0.0 for k in ALL_TASKS}
        self.lcnt = {k: 0 for k in ALL_TASKS}
        self.m.ema_update(cfg.ema)
        self.tdict = self.m.target_dict()
        for st in {id(x): x for x in (s, self.live) if x is not None}.values():
            st.st.decay_traces(cfg.s_decay)
        if self._fused:
            # Arena fused menyimpan jejak RTRL-nya sendiri. HANYA `f_st` (state maju yang
            # mengalir ke byte berikutnya) yang didekay -- PERSIS seperti jalur lama, yang
            # `decay_traces` hanya menyentuh st.layers[l].S dan BUKAN snapshot `FRec` di buf.
            # `f_st_prev` adalah snapshot backward byte berikutnya -> JANGAN didekay.
            fp.fused_decay_traces(self.f_st, self.f_d, self.f_dm, self.f_nl, cfg.s_decay)
        self.gacc, self.touched, self.nacc = {}, set(), 0
        self.stats["slow_updates"] += 1
        # Theta berubah -> aktivasi yang disimpan fused_fwd (utk byte mundur berikutnya) menjadi
        # basi. fused_bwd WAJIB menghitung ulang pass maju sekali (lihat step_fused).
        self._f_act_stale = True

    # ------------------------------------------------------------ replay & tidur
    def _replay_add(self, s):
        prio = s.seg_ce / s.seg_n
        item = {"bytes": list(s.seg), "st0": s.seg_st0, "prio": prio}
        if len(self.replay) < self.cfg.replay_cap:
            self.replay.append(item)
        else:
            j = int(np.argmin([r["prio"] for r in self.replay]))
            if prio > self.replay[j]["prio"]:
                self.replay[j] = item

    def sleep(self):
        cfg = self.cfg
        pr = np.nan_to_num(np.array([r["prio"] for r in self.replay]),
                           nan=0.0, posinf=0.0, neginf=0.0)
        k = min(cfg.sleep_segments, len(self.replay))
        pw = np.exp(pr - pr.max())
        if int((pw > 0).sum()) < k:            # underflow -> jatuh ke seragam
            pw = np.ones(len(pr))
        pw = pw / pw.sum()
        for i in self.rng.choice(len(self.replay), size=k, replace=False, p=pw):
            r = self.replay[int(i)]
            tmp = Stream(self.m, online=False)
            tmp.st = r["st0"].copy()
            for b in r["bytes"]:
                self.step(tmp, b, train=True, fast_ok=False)
        self.stats["sleeps"] += 1
        self._pl = self.pool                          # pulihkan penunjuk kolam aliran hidup

    # ------------------------------------------------------------ pemakaian
    def eval_stream(self, data, think=None):
        """CE prekuensial pada aliran data, state dingin, tanpa belajar.

        `think=None` (default) -> mode berpikir diturunkan dari `TrainOpts.think_prob`
        sehingga eval KONSISTEN dengan latihan (lihat catatan di `step`)."""
        s, out = Stream(self.m, online=False), []
        for b in data:
            c = self.step(s, b, train=False, think=think)
            if c is not None:
                out.append(c)
        return np.array(out)

    def generate(self, prompt, n, temp=0.8, think=None, seed=0):
        rng = np.random.default_rng(seed)
        s, out = Stream(self.m, online=False), []
        for b in prompt:
            self.step(s, b, train=False, think=think)
        for _ in range(n):
            e = s.buf[s.t]
            z = e.logits[e.k] / max(temp, 1e-3)
            p = np.exp(z - z.max())
            p /= p.sum()
            b = int(rng.choice(self.cfg.vocab, p=p))
            out.append(b)
            self.step(s, b, train=False, think=think)
        if self.cfg.vocab <= 256:
            return bytes(out)
        return np.asarray(out, dtype=np.int64)      # id token > 255 tidak bisa jadi bytes

    # ------------------------------------------------------------ checkpoint (kompatibel Learner)
    def save(self, path):
        m = self.m
        a = {f"p__{k}": v.data for k, v in m.params.items()}
        a.update({f"tp__{k}": v for k, v in m.tp.items()})
        for l, L in enumerate(m.layers):
            a[f"Wk__{l}"], a[f"Wv__{l}"] = L.Wk, L.Wv
        for p in self.slow:
            o, shp = self.offs[p.name]
            a[f"m__{p.name}"] = self.adam_m[o:o + p.data.size].reshape(shp).copy()
            a[f"v__{p.name}"] = self.adam_v[o:o + p.data.size].reshape(shp).copy()
        a["unc"] = np.array([self.unc[k] for k in W_TASKS])
        a["t_adam"] = np.array(self.t_adam)
        a["cfg"] = np.array(json.dumps(m.cfg.to_dict()))
        np.savez_compressed(path, **a)

    @staticmethod
    def load(path, opts=None):
        f = np.load(path, allow_pickle=False)
        c = json.loads(str(f["cfg"]))
        for k in ("horizons", "betas", "mixer_match_ks"):
            c[k] = tuple(c[k])
        model = TMT2(Config(**c))
        for k, p in model.params.items():
            if f"p__{k}" in f:                       # checkpoint lama mungkin tak punya W_out
                p.data = f[f"p__{k}"].copy()
        for k in model.tp:
            model.tp[k] = f[f"tp__{k}"].copy()
        for l, L in enumerate(model.layers):
            L.Wk, L.Wv = f[f"Wk__{l}"].copy(), f[f"Wv__{l}"].copy()
        ln = FastLearner(model, opts)
        for p in ln.slow:
            o, shp = ln.offs[p.name]
            ln.adam_m[o:o + p.data.size] = f[f"m__{p.name}"].reshape(-1)
            ln.adam_v[o:o + p.data.size] = f[f"v__{p.name}"].reshape(-1)
        ln.unc = {k: float(v) for k, v in zip(W_TASKS, f["unc"])}
        ln.t_adam = int(f["t_adam"])
        ln.tdict = model.target_dict()
        return ln
