"""Learner online TMT2 (rancangan v2): semua sinyal belajar + dua jalur optimasi.

Alur satu byte nyata b_t (Learner.step):
  1. z*_t = target-encoder(EMA)[b_t];  x_t = encoder-online(b_t)
  2. rantai berpikir: langkah nyata + hingga think_max langkah "berpikir" (masukan = p + flag,
     tanpa konsumsi byte). Kepala stop memutuskan kedalaman; state yang dibawa = kedalaman terpilih.
  3. selesaikan loss yang sudah matang (bisa terlambat):
       JEPA lokal   lapisan l, horizon h, dibuat pada langkah i=t-h; target = l2n(rata2 z*_{i+1..t})
       byte (CE)    prediksi langkah t-1 (semua kedalaman, bobot = distribusi halting)
       stop         BCE(label: gain<cost) + KL ke prior geometrik (PonderNet)
       vic / state  anti-kolaps jendela waktu / batas norma state
  4. gradien per-task dicatat di Param.tg[task]; parameter rekurensi lewat RTRL (core.rtrl_flush).
  5. JALUR CEPAT (tiap byte): parameter fast (adaptor) SGD.
     JALUR LAMBAT (tiap slow_every byte): akumulasi -> PCGrad -> clip -> AdamW -> EMA target -> S *= s_decay.
  6. replay: segmen berkejutan tinggi disimpan; tiap sleep_every byte "tidur" (jalur cepat mati).
"""
import json
from collections import deque
from dataclasses import dataclass
import numpy as np
from .autograd import (Tensor, cross_entropy, l2norm, matmul, sigmoid, tsum, tmean, tlog,
                       tsqrt, relu, reshape, concat0, transpose)
from .model import Config, TMT2, l2n_np, ln_np
from .core import StackState, stack_step, byte_logits, backward_task, rtrl_flush

W_TASKS = ("jepa", "byte", "stop")     # diseimbangkan (uncertainty + PCGrad)
F_TASKS = ("vic", "state")             # regularizer: bobot tetap (sudah termasuk di dalam loss)
ALL_TASKS = W_TASKS + F_TASKS


@dataclass
class TrainOpts:
    think_prob: float = 0.25    # peluang rantai berpikir dijalankan pada satu byte saat melatih
    vic_every: int = 8
    unc_lr: float = 0.02
    unc_clip: float = 3.0
    seed: int = 0
    use_stop: bool = True       # False: matikan loss stop (hanya ws/bs yang dilatih olehnya)
    out_lr_mult: float = 1.0    # pengali LR untuk logit_scale / byte_bias / kamus keluaran
    debug_out: bool = False     # rekam gradien parameter keluaran tiap slow_update
    track_preds: bool = False   # rekam argmax prediksi prekuensial (diagnostik)
    only_params: tuple = ()     # jika diisi: HANYA parameter ini yang diperbarui (isolasi)
    only_byte_loss: bool = False  # True: HANYA cross-entropy byte (JEPA/VIC/stop/state dimatikan)
    ngram_lam: float = 0.0      # >0: tambahkan lam*log p_bigram_online ke logits (0 = mati)
    ngram_learn: bool = False   # True: lam dipelajari sebagai parameter (task 'byte')
    ngram_order: int = 1        # orde konteks prior n-gram (1 = bigram)
    # -- kecepatan jalur lambat (default = perilaku LAMA, aman secara numerik) --
    # prenorm=True: normalisasi gradien per-parameter SEBELUM rata-rata (matematis identik
    #   karena skalar positif komutatif dengan jumlah). menurunkan biaya norm dari O(N)
    #   per elemen menjadi O(N) sekali, tetapi mengubah pembulatan float -> non-bitwise.
    prenorm: bool = False
    # norm_fast=True: ganti np.linalg.norm dengan sqrt((x*x).sum()) untuk vektor kecil.
    #   hanya lebih cepat bila ntot < ~50k (di atas itu BLAS menang); auto-pilih by ntot.
    norm_fast: bool = True
    # use_njit_adam=True: jalankan seluruh inti slow_update (norm+clip+AdamW+theta) dalam
    #   satu kernel njit. Terukur 5,4x lebih cepat dari rangkaian numpy (820->152 us @ d=32);
    #   numerik setara ~1e-15 (fastmath). False = jalur numpy lama (acuan/bitwise).
    use_njit_adam: bool = True
    # use_fused_kernel=True: jalur cepat BERBEDA yang menjalankan maju SELURUH tumpukan +
    #   kepala (dan, untuk backward, kepala + RTRL) dalam SATU panggilan njit per byte.
    #   Menghapus biaya dispatch berulang (step_fwd 42-arg + flush_rec 43-arg per lapisan).
    #   Syarat: head='linear', only_byte_loss=True, tanpa mixer/ngram/berpikir.
    #   Hasil bit-identik dengan jalur lama (uji: scripts/check_fused_fwd.py, check_fused_bwd.py).
    use_fused_kernel: bool = False


def ce_np(logits, b):
    m = logits.max()
    return float(m + np.log(np.exp(logits - m).sum()) - logits[b])


def halting(sv):
    """Distribusi halting dari peluang stop per kedalaman; kedalaman terakhir menyerap sisa."""
    n, out, rem = len(sv), [], 1.0
    for k in range(n):
        if k < n - 1:
            out.append(rem * sv[k]); rem *= 1.0 - sv[k]
        else:
            out.append(rem)
    return np.array(out)


def pcgrad(vecs, rng):
    """PCGrad (Yu dkk. 2020) atas {task: vektor gradien datar}. Untuk tiap task i (urutan acak),
    tiap task j != i (urutan acak): bila g_i . g_j < 0, buang komponen g_i searah g_j.
    Proyeksi memakai g_j ASLI (bukan yang sudah diproyeksi). Tidak mengubah masukan."""
    keys = list(vecs)
    if len(keys) < 2:
        return {k: v.copy() for k, v in vecs.items()}
    orig = {k: v.copy() for k, v in vecs.items()}
    out = {k: v.copy() for k, v in vecs.items()}
    for i in rng.permutation(len(keys)):
        ki = keys[i]
        for j in rng.permutation(len(keys)):
            kj = keys[j]
            if kj == ki:
                continue
            dot = float(out[ki] @ orig[kj])
            if dot < 0.0:
                out[ki] = out[ki] - dot / (float(orig[kj] @ orig[kj]) + 1e-12) * orig[kj]
    return out


class Entry:
    __slots__ = ("byte", "zn", "chain", "logits", "stops", "halt", "k", "chain_state", "prior",
                 "logpriors")


class Stream:
    def __init__(self, model, online=True):
        self.st = StackState(model)
        self.buf = {}                                   # indeks -> Entry
        self.t = -1
        self.zwin = deque(maxlen=model.cfg.vic_window)  # latent encoder terbaru (untuk VICReg)
        self.zwbyte = deque(maxlen=model.cfg.vic_window)  # byte yang menghasilkan tiap latent
        self.hist = deque(maxlen=64)                    # riwayat token (untuk prior n-gram orde-N)
        self.online = online
        self.seg, self.seg_ce, self.seg_n = [], 0.0, 0
        self.seg_st0 = self.st.zero_traces()


class Learner:
    def __init__(self, model, opts=None):
        self.m, self.cfg = model, model.cfg
        self.o = opts or TrainOpts()
        self.rng = np.random.default_rng(self.o.seed + 12345)
        self.tdict = model.target_dict()
        self.slow = [p for p in model.params.values() if not p.fast]
        self.fast = [p for p in model.params.values() if p.fast]
        self.adam = {p.name: [np.zeros_like(p.data), np.zeros_like(p.data)] for p in self.slow}
        self.t_adam = 0
        self.unc = {k: 0.0 for k in W_TASKS}
        self.acc, self.nacc = {}, 0
        self.lsum = {k: 0.0 for k in ALL_TASKS}
        self.lcnt = {k: 0 for k in ALL_TASKS}
        self.replay, self.live = [], None
        self.n_bytes = 0
        self.stats = {"sleeps": 0, "slow_updates": 0, "depth": np.zeros(cfg_depth(self.cfg), dtype=int)}
        self.ema_loss = {k: None for k in ALL_TASKS}
        self.offmask = 1.0 - np.eye(model.cfg.d)
        self.maxH = max(model.cfg.horizons)

    # ------------------------------------------------------------ satu byte
    def step(self, s, b, train=True, think=None, fast_ok=True):
        """Proses satu byte nyata. Mengembalikan CE prekuensial (nat) untuk b, atau None (byte pertama)."""
        m, cfg = self.m, self.cfg
        if s.online:
            self.live = s
        prev = s.buf.get(s.t)
        ce_pre = None if prev is None else ce_np(prev.logits[prev.k].data, b)

        zstar = self.tdict[b]
        x = m.encode_byte(b)
        # Mode berpikir: `think=None` -> turunkan dari konfigurasi LATIH agar eval
        # tidak menjalankan langkah berpikir yang tidak pernah dilatih (lihat
        # catatan lengkap di fast_learner.FastLearner.step).
        if think is None:
            think = (self.o.think_prob > 0.0)
        if train:
            K = cfg.think_max if (think and self.rng.random() < self.o.think_prob) else 0
        else:
            K = cfg.think_max if think else 0
        e = self._chain(s, x, zstar, K, sample=train)
        e.byte = b
        t = s.t + 1
        s.t = t
        s.buf[t] = e
        s.st = e.chain_state
        if train:
            s.zwin.append(x)
            self.stats["depth"][e.k] += 1
            self._resolve(s, t)
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
                s.seg_ce += ce_pre; s.seg_n += 1
            if len(s.seg) >= cfg.seg_len:
                if cfg.replay and s.seg_n > 0:
                    self._replay_add(s)
                s.seg, s.seg_ce, s.seg_n = [], 0.0, 0
                s.seg_st0 = s.st.zero_traces()
            if cfg.replay and self.replay and self.n_bytes % cfg.sleep_every == 0:
                self.sleep()
        return ce_pre

    # ------------------------------------------------------------ rantai berpikir
    def _chain(self, s, x, zstar, K, sample):
        m, cfg = self.m, self.cfg
        Ltop = cfg.n_layers - 1
        D = l2norm(m.dict_tensor())
        cur, chain, states, lg, sp = s.st, [], [], [], []
        xin, z = x, zstar
        for k in range(K + 1):
            recs, cur = stack_step(m, xin, cur, z)
            chain.append(recs); states.append(cur)
            p1 = recs[-1].pred(m, Ltop, 1)
            lg.append(byte_logits(m, p1, D))
            sp.append(sigmoid(matmul(Tensor(recs[-1].y.data), m.ws) + m.bs))   # probe: y di-detach
            if k == K:
                break
            if (not sample) and float(sp[-1].data) > 0.5:
                break
            xin = Tensor(ln_np(p1.data)) + m.flag       # berpikir: p kembali sebagai masukan
            z = None
        n = len(chain)
        halt = halting([float(t.data) for t in sp])
        if not sample:
            kstar = n - 1
        else:
            kstar = int(self.rng.choice(n, p=halt / halt.sum())) if n > 1 else 0
        e = Entry()
        e.byte = None
        e.zn = l2n_np(zstar)
        e.chain, e.logits, e.stops, e.halt, e.k = chain, lg, sp, halt, kstar
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

    def _resolve(self, s, t):
        m, cfg = self.m, self.cfg
        Lc = cfg.n_layers
        et = s.buf[t]
        # -- JEPA lokal, kedalaman 0, semua horizon yang matang
        jl = 0.0
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
                loss = 0.5 * tsum((l2norm(rec.pred(m, l, h)) - tgt) ** 2)
                backward_task(loss, [rec], "jepa")
                jl += float(loss.data)
        # -- JEPA pada kedalaman berpikir: horizon 1 terhadap z*_t
        if t - 1 in s.buf:
            ei = s.buf[t - 1]
            for k in range(1, len(ei.chain)):
                for l in range(Lc):
                    rec = ei.chain[k][l]
                    loss = 0.5 * tsum((l2norm(rec.pred(m, l, 1)) - et.zn) ** 2)
                    backward_task(loss, [rec], "jepa")
                    jl += float(loss.data)
        if jl > 0:
            self._note("jepa", jl)
        # -- byte + stop (prediksi langkah t-1 vs byte t)
        if t - 1 in s.buf and et.byte is not None:
            ei, b = s.buf[t - 1], et.byte
            n = len(ei.chain)
            ces, tip, tops = [], 0.0, []
            for k in range(n):
                ce = cross_entropy(ei.logits[k], b)
                ces.append(float(ce.data))
                if ei.halt[k] > 1e-6:
                    tip = tip + ce * float(ei.halt[k]); tops.append(ei.chain[k][-1])
            backward_task(tip, tops, "byte")
            self._note("byte", float(np.dot(ei.halt, ces)))
            if n > 1:
                terms = []
                for k in range(n - 1):
                    lab = 1.0 if (ces[k] - ces[k + 1]) < cfg.think_cost else 0.0
                    sg = ei.stops[k]
                    terms.append(-(lab * tlog(sg + 1e-9) + (1.0 - lab) * tlog(1.0 - sg + 1e-9)))
                rem, kl, rho = Tensor(1.0), 0.0, cfg.stop_prior
                for k in range(n):
                    if k < n - 1:
                        pk = rem * ei.stops[k]; rem = rem * (1.0 - ei.stops[k]); qk = rho * (1 - rho) ** k
                    else:
                        pk = rem; qk = (1 - rho) ** k
                    kl = kl + pk * (tlog(pk + 1e-9) - float(np.log(qk)))
                loss = sum(terms) + cfg.stop_kl * kl
                backward_task(loss, [], "stop")
                self._note("stop", float(loss.data))

    def _state_cap(self, e):
        cfg = self.cfg
        tot = 0.0
        for rec in e.chain[e.k]:
            h = rec.h_leaf.data
            ex = np.maximum(np.abs(h) - cfg.state_cap, 0.0)
            if ex.any():
                rec.h_leaf.grad = cfg.state_w * 2.0 * ex * np.sign(h) / cfg.d
                rtrl_flush(rec, "state")
                tot += cfg.state_w * float((ex * ex).mean())
        if tot > 0:
            self._note("state", tot)

    def _vic(self, s):
        cfg = self.cfg
        n = len(s.zwin)
        if n < 8 or s.t % self.o.vic_every:
            return
        d = cfg.d
        Z = concat0([reshape(z, (1, d)) for z in s.zwin])
        Zc = Z - tmean(Z, axis=0, keepdims=True)
        std = tsqrt(tmean(Zc * Zc, axis=0) + 1e-4)
        lv = tmean(relu(cfg.vic_gamma - std))
        cov = matmul(transpose(Zc), Zc) / float(n - 1)
        lc = tsum((cov * self.offmask) ** 2) / float(d)
        loss = (lv + cfg.vic_cov * lc) * cfg.vic_w
        backward_task(loss, [], "vic")
        self._note("vic", float(loss.data))

    # ------------------------------------------------------------ optimasi
    def _weight(self, task):
        return float(np.exp(-self.unc[task])) if (task in W_TASKS and self.cfg.uncertainty) else 1.0

    def _consume(self, s, fast_ok):
        """Ambil gradien per-task langkah ini: jalur cepat (SGD) + akumulasi untuk jalur lambat."""
        cfg = self.cfg
        for p in self.m.params.values():
            if not p.tg:
                continue
            if p.fast:
                if fast_ok:
                    g = sum(self._weight(k) * v for k, v in p.tg.items())
                    n = np.linalg.norm(g)
                    if n > 1.0:
                        g = g / n
                    p.data = p.data * (1.0 - cfg.fast_decay) - cfg.fast_lr * g   # bukan in-place
            else:
                for k, v in p.tg.items():
                    d = self.acc.setdefault(k, {})
                    d[p.name] = v if p.name not in d else d[p.name] + v
            p.tg = {}
        self.nacc += 1
        if self.nacc >= cfg.slow_every:
            self.slow_update(s)

    def slow_update(self, s=None):
        cfg = self.cfg
        if not self.acc:
            self.nacc = 0
            return
        K = float(self.nacc)
        names = [p.name for p in self.slow]
        sizes = {p.name: p.data.size for p in self.slow}

        def flat(task):
            g = self.acc.get(task, {})
            return np.concatenate([(g[n] / K).ravel() if n in g else np.zeros(sizes[n]) for n in names])

        wt = [k for k in W_TASKS if k in self.acc]
        vecs = {k: self._weight(k) * flat(k) for k in wt}
        if cfg.pcgrad and len(wt) > 1:
            vecs = pcgrad(vecs, self.rng)
        total = sum(vecs.values()) if vecs else np.zeros(sum(sizes.values()))
        for k in F_TASKS:
            if k in self.acc:
                total = total + flat(k)
        gn = float(np.linalg.norm(total))
        if gn > cfg.clip:
            total = total * (cfg.clip / gn)
        # AdamW
        self.t_adam += 1
        lr = cfg.lr * min(1.0, self.t_adam / max(cfg.warmup, 1))
        b1, b2 = cfg.betas
        i0 = 0
        touched = {n for t in self.acc.values() for n in t}
        for p in self.slow:
            g = total[i0:i0 + p.data.size].reshape(p.data.shape)
            i0 += p.data.size
            if p.name not in touched:
                continue
            mm, vv = self.adam[p.name]
            mm = b1 * mm + (1 - b1) * g
            vv = b2 * vv + (1 - b2) * g * g
            self.adam[p.name] = [mm, vv]
            upd = (mm / (1 - b1 ** self.t_adam)) / (np.sqrt(vv / (1 - b2 ** self.t_adam)) + 1e-8)
            decay = (1.0 - lr * cfg.wd) if p.data.ndim >= 2 else 1.0
            p.data = p.data * decay - lr * upd
        # uncertainty (Kendall): L = exp(-s) Lbar + s
        if cfg.uncertainty:
            for k in W_TASKS:
                if self.lcnt[k]:
                    Lbar = self.lsum[k] / self.lcnt[k]
                    sk = self.unc[k] - self.o.unc_lr * (1.0 - np.exp(-self.unc[k]) * Lbar)
                    self.unc[k] = float(np.clip(sk, -self.o.unc_clip, self.o.unc_clip))
        self.lsum = {k: 0.0 for k in ALL_TASKS}
        self.lcnt = {k: 0 for k in ALL_TASKS}
        # EMA encoder target, jejak RTRL meluruh (bobot sudah berubah)
        self.m.ema_update(cfg.ema)
        self.tdict = self.m.target_dict()
        for st in {id(x): x for x in (s, self.live) if x is not None}.values():
            st.st.decay_traces(cfg.s_decay)
        self.acc, self.nacc = {}, 0
        self.stats["slow_updates"] += 1

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
        pw /= pw.sum()
        for i in self.rng.choice(len(self.replay), size=k, replace=False, p=pw):
            r = self.replay[int(i)]
            tmp = Stream(self.m, online=False)
            tmp.st = r["st0"].copy()
            for j, b in enumerate(r["bytes"]):
                self.step(tmp, b, train=True, fast_ok=False)
        self.stats["sleeps"] += 1

    # ------------------------------------------------------------ pemakaian
    def eval_stream(self, data, think=None):
        """CE prekuensial (nat) per byte TANPA belajar.

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
            z = e.logits[e.k].data / max(temp, 1e-3)
            p = np.exp(z - z.max()); p /= p.sum()
            b = int(rng.choice(self.cfg.vocab, p=p))
            out.append(b)
            self.step(s, b, train=False, think=think)
        return bytes(out)

    # ------------------------------------------------------------ checkpoint
    def save(self, path):
        m = self.m
        a = {f"p__{k}": v.data for k, v in m.params.items()}
        a.update({f"tp__{k}": v for k, v in m.tp.items()})
        for l, L in enumerate(m.layers):
            a[f"Wk__{l}"], a[f"Wv__{l}"] = L.Wk, L.Wv
        for k, (x, y) in self.adam.items():
            a[f"m__{k}"], a[f"v__{k}"] = x, y
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
        ln = Learner(model, opts)
        for k in ln.adam:
            ln.adam[k] = [f[f"m__{k}"].copy(), f[f"v__{k}"].copy()]
        ln.unc = {k: float(v) for k, v in zip(W_TASKS, f["unc"])}
        ln.t_adam = int(f["t_adam"])
        ln.tdict = model.target_dict()
        return ln


def cfg_depth(cfg):
    return cfg.think_max + 1
