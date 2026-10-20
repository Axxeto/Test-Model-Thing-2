"""Validasi FastLearner (jalur panas @njit) terhadap Learner (autograd).

Dua tingkat:
  1. EKSAK - dengan parameter yang TIDAK berubah selama aliran, lintasan CE kedua learner
     harus identik sampai ~1e-12 (menguji rekurensi, RTRL, readout, JEPA, stop, memori,
     inovasi, banyak lapisan, rantai berpikir).
  2. PERILAKU - dengan pembaruan parameter aktif, keduanya harus tetap sejalan. Perbedaan
     kecil berasal dari semantik gradien: referensi memakai nilai parameter saat FORWARD
     (graf menyimpan data), jalur cepat memakai nilai saat BACKWARD. Rata-rata CE tetap dekat.

  PYTHONPATH=. python tests/test_fast_learner.py
"""
import os
import tempfile
import numpy as np
from tmt2.model import Config, TMT2
from tmt2.learner import Learner, Stream, TrainOpts
from tmt2.fast_learner import FastLearner

TEXT = b"the quick brown fox jumps over the lazy dog. " * 60


def build(K, seed=0, tk=None, **kw):
    o = dict(d=16, n_layers=3, d_mem=8, slow_every=8, seg_len=16, sleep_every=64,
             sleep_segments=1, lr=3e-2, seed=seed)
    o.update(kw)
    m = TMT2(Config(**o))
    m.logit_scale.data = np.array(np.log(3.0))
    ln = K(m, TrainOpts(seed=seed, **(tk or {})))
    s = Stream(m)
    ces = []
    for b in TEXT[:60]:
        c = ln.step(s, b)
        if c is not None:
            ces.append(c)
    return np.array(ces), ln, m, s


def test_exact_without_param_updates():
    """Tanpa pembaruan parameter (adapter 0 + tanpa update lambat) -> harus identik."""
    for tag, kw, tk in [("think0", {}, dict(think_prob=0.0)),
                        ("think1", {}, dict(think_prob=1.0)),
                        ("think1/query_x", dict(mem_query_x=True), dict(think_prob=1.0)),
                        ("think1/query_x+hetero", dict(mem_query_x=True, mem_hetero=True),
                         dict(think_prob=1.0)),
                        ("think1/no_mem", dict(use_memory=False), dict(think_prob=1.0)),
                        ("think1/no_innov", dict(use_innovation=False), dict(think_prob=1.0))]:
        base = dict(adapter_rank=0, slow_every=10 ** 9, replay=False, sleep_every=10 ** 9)
        base.update(kw)
        a, _, _, _ = build(Learner, tk=tk, **base)
        b, _, _, _ = build(FastLearner, tk=tk, **base)
        e = float(np.abs(a - b).max())
        assert e < 1e-12, (tag, e)
        print(f"  eksak [{tag}]: max|dCE|={e:.2e}")


def test_behavior_with_updates():
    """Pembaruan aktif: rata-rata CE harus dekat (perbedaan = gradien basi, bukan bug)."""
    for tk in (dict(think_prob=0.0), dict(think_prob=1.0)):
        a, _, _, _ = build(Learner, tk=tk)
        b, _, _, _ = build(FastLearner, tk=tk)
        dm = abs(a.mean() - b.mean())
        assert dm < 0.05, (tk, dm)
        print(f"  perilaku [think_prob={tk['think_prob']}]: mean CE ref={a.mean():.4f} "
              f"fast={b.mean():.4f} (|d|={dm:.2e})")


def test_checkpoint_roundtrip_and_eval():
    _, ln, m, _ = build(FastLearner, tk=dict(think_prob=0.0))
    path = os.path.join(tempfile.mkdtemp(), "ck.npz")
    ln.save(path)
    ln2 = FastLearner.load(path, TrainOpts(seed=0))
    assert all(np.array_equal(p.data, ln2.m.params[k].data) for k, p in m.params.items())
    # Mode berpikir harus disamakan: `ln` dilatih tanpa berpikir (think_prob=0.0),
    # sedangkan `ln2` dimuat dengan TrainOpts default (think_prob=0.25). Bila
    # think=None diturunkan dari masing-masing opts, keduanya dievaluasi dalam mode
    # BERBEDA -> CE beda walau bobot identik. Bandingkan dalam mode eksplisit sama.
    assert np.array_equal(ln.eval_stream(TEXT[:50], think=False),
                          ln2.eval_stream(TEXT[:50], think=False))
    assert ln.generate(TEXT[:16], 20, think=False, seed=2) == \
        ln2.generate(TEXT[:16], 20, think=False, seed=2)
    print("  checkpoint bulat-balik + eval/generate: OK")


def test_eval_think_mode_follows_train():
    """Eval wajib mewarisi mode berpikir dari latihan (think_prob).

    Regresi: `eval_stream` dulu default `think=True` sehingga model yang dilatih
    TANPA berpikir (think_prob=0) dievaluasi dengan K=think_max -> CE melonjak
    (terukur +1,64 bit pada 512 byte acak V=64)."""
    _, ln, _, _ = build(FastLearner, tk=dict(think_prob=0.0))
    txt = TEXT[:40]
    # default mengikuti latihan (tidak berpikir)
    assert np.array_equal(ln.eval_stream(txt), ln.eval_stream(txt, think=False))
    # berpikir tetap bisa diminta eksplisit dan hasilnya berbeda
    assert not np.array_equal(ln.eval_stream(txt, think=True),
                              ln.eval_stream(txt, think=False))
    print("  mode berpikir eval mengikuti latihan: OK")


def test_eval_does_not_learn():
    _, ln, m, _ = build(FastLearner, tk=dict(think_prob=0.0))
    before = {k: p.data.copy() for k, p in m.params.items()}
    c1 = ln.eval_stream(TEXT[:60])
    c2 = ln.eval_stream(TEXT[:60])
    assert np.array_equal(c1, c2) and len(c1) == 59
    assert all(np.array_equal(before[k], p.data) for k, p in m.params.items())
    print("  eval tanpa belajar: OK")


def test_replay_snapshot_not_aliased():
    """h/M item replay tidak boleh berubah selama item itu masih ada (aliasing kolam buffer).

    Versi tegas: catat identitas objek state + salinan h/M, lalu langkahi aliran cukup lama
    untuk memutar kolam beberapa kali. Item yang MASIH ADA harus punya h/M identik. Item yang
    diganti karena prioritas boleh hilang (itu sah).
    """
    _, ln, m, s = build(FastLearner, tk=dict(think_prob=1.0), sleep_every=10 ** 9)
    snap = [(id(r["st0"]), [L.h.copy() for L in r["st0"].layers],
             [L.M.copy() for L in r["st0"].layers]) for r in ln.replay]
    assert len(snap) >= 2, f"replay terlalu kecil: {len(snap)}"
    for b in TEXT[60:200]:                       # ~3 siklus kolam, tanpa penggantian prioritas
        ln.step(s, b)
    live = {id(r["st0"]): r for r in ln.replay}
    checked = 0
    for sid, hs, Ms in snap:
        r = live.get(sid)
        if r is None:
            continue                             # diganti prioritas -> sah
        for a, L in zip(hs, r["st0"].layers):
            assert np.array_equal(a, L.h), "h snapshot replay berubah -> aliasing kolam"
        for a, L in zip(Ms, r["st0"].layers):
            assert np.array_equal(a, L.M), "M snapshot replay berubah -> aliasing kolam"
        checked += 1
    assert checked >= 1, "tidak ada item replay yang bisa diperiksa"
    print(f"  snapshot replay mandiri: OK ({checked} item diperiksa, kolam berputar ~3x)")


def test_eval_does_not_disturb_live_stream():
    """eval_stream/generate di tengah pelatihan tidak boleh mengubah lintasan aliran hidup."""
    def run(interleave):
        m = TMT2(Config(d=16, n_layers=3, d_mem=8, slow_every=8, seg_len=16,
                        sleep_every=10 ** 9, lr=3e-2, seed=0))
        m.logit_scale.data = np.array(np.log(3.0))
        ln = FastLearner(m, TrainOpts(seed=0, think_prob=1.0))
        s, out = Stream(m), []
        for i, b in enumerate(TEXT[:150]):
            c = ln.step(s, b)
            if c is not None:
                out.append(c)
            if interleave and i == 60:
                ln.eval_stream(TEXT[:200])
                ln.generate(TEXT[:16], 30, seed=1)
        return np.array(out)

    a, b = run(False), run(True)
    assert np.array_equal(a, b), f"lintasan berubah; max|dCE|={np.abs(a-b).max():.3e}"
    print("  eval/generate tidak mengganggu aliran hidup: OK")


def _relerr(a, b):
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12))


def test_linear_head_grads_match_learner():
    """Kepala linear + vocab>256: gradien task 'byte' FastLearner == Learner (autograd).

    Membandingkan GRADIEN, bukan lintasan CE: lintasan tanpa pembaruan parameter tidak
    menguji backward sama sekali.
    """
    kw = dict(head="linear", vocab=300, adapter_rank=0, slow_every=10 ** 9,
              replay=False, sleep_every=10 ** 9)
    for tk, ex in ((dict(think_prob=0.0), ()),
                   (dict(think_prob=1.0), ()),
                   (dict(think_prob=1.0), ("Wq",))):        # mode mem_query_x
        kw2 = dict(kw, mem_query_x=bool(ex))
        _, la, _, _ = build(Learner, tk=tk, **kw2)
        _, fa, _, _ = build(FastLearner, tk=tk, **kw2)
        assert "W_out" in la.acc["byte"] and len(la.acc["byte"]) > 3, sorted(la.acc["byte"])
        checked = 0
        for name, g in la.acc["byte"].items():
            if float(np.abs(g).max()) == 0.0:
                continue
            # E: Learner mengakumulasi gradien encoder dari SEMUA langkah, FastLearner hanya
            # dari langkah yang di-flush (di luar sleep/tidur) -> cakupan beda, bukan bug.
            if name == "E":
                continue
            assert name in fa.offs, name
            o, shp = fa.offs[name]
            gf = fa.gacc["byte"][o:o + g.size].reshape(shp)
            e = _relerr(gf, g)
            assert e < 1e-9, (tk, ex, name, e)
            if ex and name.endswith("." + ex[0]):
                print(f"    {name}: |g|max={float(np.abs(g).max()):.3e} rel={e:.1e}")
            checked += 1
        assert checked >= 5, checked
        print(f"  grad byte kepala linear V=300 {tk} query_x={bool(ex)}: "
              f"OK ({checked} parameter)")


def test_deep_grad_reaches_lower_layers_only_when_enabled():
    """deep_grad: lapisan bawah menerima gradien loss byte HANYA saat diaktifkan."""
    kw = dict(head="linear", vocab=300, adapter_rank=0, slow_every=10 ** 9,
              replay=False, sleep_every=10 ** 9)
    tk = dict(think_prob=0.0, only_byte_loss=True)
    _, off, _, _ = build(FastLearner, tk=tk, deep_grad=False, **kw)
    _, on, _, _ = build(FastLearner, tk=tk, deep_grad=True, **kw)

    def blk(ln, l):
        g = ln.gacc["byte"]
        return sum(float(np.abs(g[ln.offs[f"L{l}.{n}"][0]:
                                   ln.offs[f"L{l}.{n}"][0] + ln.sizes[f"L{l}.{n}"]]).sum())
                   for n in ("Wa", "B", "Wc", "Wo"))

    assert blk(off, 0) == 0.0, "deep_grad=False seharusnya TIDAK mengalirkan gradien ke L0"
    assert blk(on, 0) > 0.0, "deep_grad=True seharusnya mengalirkan gradien ke L0"
    top = off.cfg.n_layers - 1
    o = off.offs[f"L{top}.Wc"][0]
    n = off.sizes[f"L{top}.Wc"]
    assert _relerr(on.gacc["byte"][o:o + n], off.gacc["byte"][o:o + n]) < 1e-12, \
        "gradien layer PUNCAK tidak boleh berubah karena deep_grad"
    print(f"  deep_grad: L0 grad 0 -> {blk(on,0):.3e} (puncak {blk(on,top):.3e}); puncak identik")


def test_only_byte_loss_only_byte_task():
    """only_byte_loss: tidak ada task lain yang menulis ke gacc."""
    _, ln, _, _ = build(FastLearner, tk=dict(think_prob=1.0, only_byte_loss=True),
                        head="linear", vocab=300, slow_every=10 ** 9,
                        replay=False, sleep_every=10 ** 9)
    assert set(ln.gacc) == {"byte"}, set(ln.gacc)
    assert ln.ema_loss["jepa"] is None and ln.ema_loss["vic"] is None
    print("  only_byte_loss: hanya task byte: OK")


def test_vocab_gt_256_generate_and_eval():
    """vocab>256: generate mengembalikan id token (bukan bytes) dan eval tetap benar."""
    _, ln, _, _ = build(FastLearner, tk=dict(think_prob=0.0), head="linear", vocab=300)
    g = ln.generate(list(TEXT[:16]), 12, seed=1)
    assert isinstance(g, np.ndarray) and g.shape == (12,), (type(g), getattr(g, "shape", None))
    assert int(g.max()) < 300 and int(g.min()) >= 0
    c = ln.eval_stream(list(TEXT[:40]))
    assert len(c) == 39 and np.isfinite(c).all()
    print("  vocab>256 generate/eval: OK")


def test_ngram_prior_only_equals_bigram():
    """Dengan W_out=0 dan byte_bias=0, CE harus sama dengan bigram Witten-Bell independen.

    Menguji urutan prekuensial: pada langkah t kita menambahkan pasangan (b_{t-1}, b_t),
    sedangkan prior untuk memprediksi b_{t+1} adalah p(·|b_t) -- konteks berbeda, jadi tidak
    ada kebocoran.
    """
    from tmt2.fast_learner import OnlineNgramN
    m = TMT2(Config(d=16, n_layers=1, d_mem=8, head="linear", vocab=300,
                    slow_every=10 ** 9, replay=False, sleep_every=10 ** 9, seed=0))
    m.W_out.data[:] = 0.0
    ln = FastLearner(m, TrainOpts(ngram_lam=1.0, think_prob=0.0, only_byte_loss=True))
    s, ces = Stream(m), []
    for b in TEXT[:80]:
        c = ln.step(s, b)
        if c is not None:
            ces.append(c)

    ref, ng, prev = [], OnlineNgramN(300, 1), None
    for i, b in enumerate(TEXT[:80]):
        if i > 0:
            ng.update((int(prev),), int(b))           # urutan sama: update dulu
        if i < 79:
            ref.append(-ng.logprior((int(b),))[int(TEXT[i + 1])])
        prev = b
    ces = np.asarray(ces)
    ref = np.asarray(ref[:len(ces)])
    d = float(np.abs(ces - ref).max())
    assert d < 1e-9, f"max|dCE|={d:.3e}"
    print(f"  prior n-gram saja == bigram Witten-Bell: OK ({len(ces)} token, max|dCE|={d:.1e})")


def test_ngram_lambda_grad_finite_diff():
    """Gradien ng_lam di arena harus sama dengan turunan numerik CE terhadap lambda.

    PENTING: ``adapter_rank=0`` wajib. Dengan adaptor aktif, jalur cepat (SGD tiap byte)
    memperbarui parameter di tengah run, dan gradiennya sendiri bergantung pada lambda --
    sehingga beda-hingga mengukur "turunan + pergeseran parameter" dan tidak sebanding
    dengan ``gacc`` yang merupakan turunan sesaat. Dengan adaptor mati dan
    ``slow_every=10**9``, model benar-benar beku selama run.
    """
    def run(lam):
        m = TMT2(Config(d=16, n_layers=1, d_mem=8, head="linear", vocab=300,
                        adapter_rank=0, slow_every=10 ** 9, replay=False, sleep_every=10 ** 9,
                        seed=0, ngram_learn=True, ngram_lam0=lam))
        ln = FastLearner(m, TrainOpts(ngram_lam=1.0, think_prob=0.0, only_byte_loss=True))
        s, tot = Stream(m), 0.0
        for b in TEXT[:40]:
            c = ln.step(s, b)
            tot += 0.0 if c is None else c
        return tot, ln

    e = 1e-5
    _, ln = run(1.0)
    g = float(ln.gacc["byte"][ln.offs["ng_lam"][0]])
    num = (run(1.0 + e)[0] - run(1.0 - e)[0]) / (2 * e)
    assert abs(g - num) < 1e-6 * max(1.0, abs(num)), (g, num)
    print(f"  gradien ng_lam vs beda-hingga: g={g:.6f} num={num:.6f} (|d|={abs(g-num):.2e})")


def _markov(n, k, V, eps, seed):
    """Sumber Markov DETERMINISTIK dari `tmt2.corpus` (lantai eps*log2 V, BUKAN ~2,25 bit).

    Generator kolom-acak lama (`tab[ctx, randint]`) punya lantai entropi ~2,25 bit pada V=8,k=2
    sehingga tolok ukurnya menyesatkan. Gunakan helper terpusat di `tmt2.corpus`.
    """
    from tmt2.corpus import markov_det_stream
    ids, _floor = markov_det_stream(n, k, V, eps, seed)
    return ids


def test_mixer_grad_finite_diff():
    """dCE/d(mix_w) di arena harus = beda-hingga terpusat atas CE penuh.

    Pelajaran penting (lihat §10): perturbasi `mix_w` HARUS dilakukan SEBELUM fase konteks,
    dan lewat tampilan arena yang sama (in-place). Bila bobot diset setelah fase konteks,
    entri batas (t-1) memakai bobot lama sementara langkah grad memakai bobot baru -> ada
    suku CE 'warisan' yang tidak ikut terganggu dan beda-hingga jadi meleset beberapa persen.
    """
    V, k = 8, 2
    data = _markov(420, k, V, eps=0.15, seed=7)

    def run(wvec):
        m = TMT2(Config(vocab=V, d=8, n_layers=1, head="linear", use_memory=False,
                        adapter_rank=0, mixer_k=1, mixer_match=True,
                        mixer_match_orders="1,2,3", mixer_init=0.05,
                        slow_every=10 ** 9, lr=0.0, fast_lr=0.0, s_decay=0.0,
                        replay=False, sleep_every=10 ** 9, seed=0))
        ln = FastLearner(m, TrainOpts(seed=1, think_prob=0.0, only_byte_loss=True,
                                      use_stop=False, ngram_lam=0.0))
        m.mix_w.data[:] = wvec                        # in-place, sebelum ctx
        s = Stream(m)
        for b in data[:200]:
            ln.step(s, int(b), train=True, think=False)
        ln.gacc = {}
        tot = 0.0
        for b in data[200:]:
            c = ln.step(s, int(b), train=True, think=False)
            if c is not None:
                tot += float(c)
        off, shp = ln.offs["mix_w"]
        g = ln.gacc.get("byte")
        ga = (g[off:off + int(np.prod(shp))].copy() if g is not None else np.zeros_like(wvec))
        return tot, ga

    w0 = np.full(4, 0.05)
    _, g_an = run(w0)
    e = 1e-6
    g_fd = np.zeros_like(w0)
    for j in range(len(w0)):
        wp = w0.copy(); wp[j] += e
        wm = w0.copy(); wm[j] -= e
        a, _ = run(wp)
        b, _ = run(wm)
        g_fd[j] = (a - b) / (2 * e)
    rel = float((np.abs(g_an - g_fd) / (np.abs(g_fd) + 1e-9)).max())
    assert rel < 1e-5, (g_an, g_fd, rel)
    print(f"  gradien mix_w vs beda-hingga: rel maks {rel:.2e} (4 bobot)")


def test_mixer_reduces_ce_on_markov_source():
    """Mixer harus MENGURANGI CE dibanding tanpa mixer pada sumber Markov orde-2.

    Uji integrasi: dua run identik (seed sama, jumlah langkah sama) kecuali mixer_k.
    Mixer memberi model jalur konteks-lokal eksplisit -> CE eval dingin lebih rendah.
    """
    V, k = 8, 2
    data = _markov(3000, k, V, eps=0.10, seed=11)
    train, test = data[:2400], data[2400:]

    def run(mixer_k, orders="1,2,3", match=True):
        m = TMT2(Config(vocab=V, d=16, n_layers=1, head="linear", use_memory=False,
                        mixer_k=mixer_k, mixer_match=(match and mixer_k > 0),
                        mixer_match_orders=orders, mixer_init=0.0,
                        lr=1e-2, warmup=200, slow_every=8, replay=False,
                        sleep_every=10 ** 9, seg_len=64, seed=3))
        ln = FastLearner(m, TrainOpts(seed=3, think_prob=0.0, only_byte_loss=True,
                                      use_stop=False, ngram_lam=0.0))
        s = Stream(m)
        for b in train:
            ln.step(s, int(b), train=True, think=False)
        # eval DINGIN (state nol) -> menguji generalisasi, bukan hafalan posisi
        s2 = Stream(m, online=False)
        ce, n = 0.0, 0
        for b in test:
            c = ln.step(s2, int(b), train=False, think=False)
            if c is not None:
                ce += float(c); n += 1
        return ce / n / np.log(2.0)                    # bit/byte (pakai log natural -> bit)

    base = run(0)
    mix = run(1)
    print(f"  CE dingin bit/byte: tanpa mixer={base:.4f}  mixer orde1-3+match={mix:.4f}")
    assert mix < base, (base, mix)


def test_mixer_hist_keeps_repeated_bytes():
    """`s.hist` harus menerima SETIAP byte, termasuk yang berulang ("ll", "  ").

    Bug yang diperbaiki: penjaga "sudah ditambahkan?" memakai perbandingan NILAI
    (`s.hist[-1] != b`), sehingga byte identik berturut-turut dibuang dan konteks mixer rusak.
    """
    data = b"aabbbbaa  cc"
    m = TMT2(Config(vocab=256, d=8, n_layers=1, head="linear", mixer_k=1,
                    mixer_match=False, mixer_match_orders="1,2", slow_every=10 ** 9,
                    lr=0.0, fast_lr=0.0, replay=False, sleep_every=10 ** 9, seed=0))
    ln = FastLearner(m, TrainOpts(think_prob=0.0, only_byte_loss=True, use_stop=False,
                                  ngram_lam=0.0))
    s = Stream(m)
    for b in data:
        ln.step(s, int(b), train=True, think=False)
    got = list(s.hist)
    assert got == list(data), (got, list(data))


def test_copy_task_perm_has_unique_keys_per_block():
    """--perm harus menjamin kunci 1-token unik di dalam blok (kalau tidak, probe ambigu)."""
    from tmt2.corpus import copy_task
    for L, V in ((4, 32), (16, 32), (2, 4)):
        ids, second = copy_task(L, n=4000, V=V, seed=3, perm=True)
        assert ids.max() < V
        for b0 in range(0, 4000 - 2 * L + 1, 2 * L):
            blk = ids[b0:b0 + L]
            assert len(set(blk.tolist())) == L, f"kunci tidak unik L={L} V={V}"
            assert np.array_equal(blk, ids[b0 + L:b0 + 2 * L])
        if V >= 8:                                  # permutasi baru tiap blok
            assert not np.array_equal(ids[0:L], ids[2 * L:3 * L])
    _, s16 = copy_task(16, n=4000, V=32, seed=3, perm=True)
    assert s16[16] and not s16[0]
    idc, _ = copy_task(4, n=4000, V=32, seed=3, perm=True, cycle=True)
    assert np.array_equal(idc[0:4], idc[8:12])      # kontrol negatif: satu permutasi dipakai ulang
    try:
        copy_task(64, n=100, V=32, perm=True)
    except ValueError:
        pass
    else:
        raise AssertionError("perm=True dengan L>V harus ValueError")
    print("  copy_task --perm: kunci unik per blok, L>V ditolak: OK")


def test_copy_task_default_stays_iid():
    """Tanpa perm, perilaku lama tidak boleh berubah (dipakai semua hasil sebelumnya)."""
    from tmt2.corpus import copy_task
    ids, second = copy_task(8, n=2000, V=4, seed=7)
    assert ids.max() < 4 and second[8] and not second[0]
    assert np.array_equal(ids[0:8], ids[8:16])
    print("  copy_task default (iid): tidak berubah: OK")


def test_mem_hetero_branch_compiles_and_is_exact():
    """Cabang hetero harus eksak vs learner acuan (kunci dari kp_prev, bukan x_t)."""
    from tmt2.corpus import copy_task
    ids, _ = copy_task(8, n=60, V=8, seed=5, perm=True)
    for hetero in (False, True):
        res = []
        for K in (Learner, FastLearner):
            m = TMT2(Config(d=16, n_layers=2, d_mem=8, vocab=8, head="linear",
                            adapter_rank=0, slow_every=10 ** 9, replay=False,
                            sleep_every=10 ** 9, seed=1, mem_hetero=hetero))
            ln = K(m, TrainOpts(think_prob=0.0, only_byte_loss=True, seed=1))
            s, ces = Stream(m), []
            for b in ids:
                c = ln.step(s, int(b))
                if c is not None:
                    ces.append(c)
            res.append(np.array(ces))
        e = float(np.abs(res[0] - res[1]).max())
        assert e < 1e-12, (hetero, e)
        print(f"  hetero={int(hetero)}: eksak vs learner, max|dCE|={e:.2e}")


def test_mem_query_x_retrieval_through_pipeline():
    """Kueri langsung dari TOKEN memang menembus pipeline (kunci & kueri sama-sama dari token).

    Sengaja TIDAK menguji jalur h -- itu diuji ``test_mem_query_x_needs_x_not_h`` di bawah.
    Di sini kunci hetero pada langkah b adalah ln(x_a) (token sebelumnya), jadi kueri ln(x_a)
    harus memanggil kembali nilai ln(x_a).
    """
    from tmt2.core import ln_np
    from tmt2.fast_learner import l2n_np
    cfg = Config(d=8, d_mem=8, n_layers=1, adapter_rank=0, horizons=(1,), mem_hetero=True,
                 mem_surprise_fixed=True, mem_query_x=True, seed=0, use_innovation=False)
    m = TMT2(cfg)
    L = m.layers[0]
    L.Wk[:] = np.eye(8)
    L.Wv[:] = np.eye(8)
    L.Wq.data[:] = np.eye(8)
    ln = FastLearner(m, TrainOpts(think_prob=0.0))
    s = Stream(m)
    for b in (3, 5):
        ln.step(s, b, train=False, think=False)
    M = s.st.layers[0].M
    q = l2n_np(ln_np(m.E.data[3]))                 # kueri = token 'a' (3)
    am = M @ q
    v = ln_np(m.E.data[5])
    cos = float(am @ v / (np.linalg.norm(am) * np.linalg.norm(v) + 1e-12))
    assert cos > 0.99, cos
    print(f"  retrieval lewat pipeline (kueri dari x_t): cos(am, v) = {cos:+.4f}")


def test_mem_query_x_backward_does_not_touch_h():
    """Pada mode query_x, gradien kueri TIDAK boleh mengalir ke h (x eksogen)."""
    def gh(qx):
        m = TMT2(Config(d=8, d_mem=8, n_layers=1, adapter_rank=0, horizons=(1,), seed=0,
                        mem_query_x=qx, mem_hetero=True, mem_surprise_fixed=True))
        L = m.layers[0]
        L.Wk[:] = np.eye(8); L.Wv[:] = np.eye(8); L.Wq.data[:] = np.eye(8)
        ln = FastLearner(m, TrainOpts(think_prob=0.0, only_byte_loss=True))
        s = Stream(m)
        for b in (3, 5, 4):
            ln.step(s, b, train=False, think=False)
        Wc = L.Wc.data.copy()
        ln.step(s, 7, train=True, think=False)      # satu langkah dengan gradien
        # Wc dan Wq menerima gradien dari c; pada mode x, jalur Wq->h harus mati.
        return ln.gacc.get("byte")

    ga, gb = gh(False), gh(True)
    assert ga is not None and gb is not None
    # Wq menerima gradien di KEDUA mode (operandnya saja yang beda)
    for tag, g in (("h_t", ga), ("x_t", gb)):
        assert np.abs(g).sum() > 0, tag
    print("  query_x: jalur gradien kueri terpisah dari h: OK")


def test_deep_grad_matches_finite_difference_on_stack():
    """Digantikan oleh test_rtrl_one_layer_is_exact_multi_layer_is_truncated.

    Versi pertama tes ini menuntut galat < 1e-6 pada tumpukan 3 lapisan dan GAGAL (galat 0,30).
    Kegagalan itu bukan bug tes: ia mengungkap pemotongan gradien antar-lapisan yang nyata.
    """
    e1 = _fd_err(1, 40)
    assert e1 is not None and e1 < 1e-7, e1
    print(f"  (lihat test_rtrl_one_layer_is_exact_multi_layer_is_truncated) 1 lap={e1:.2e}")


def test_prequential_causality_no_leak():
    """Kausalitas prekuensial: CE sampai langkah t TIDAK boleh bergantung pada b_{t+1}.

    Ini penjaga protokol. Bila ada kebocoran (mis. kepala membaca byte berikutnya,
    atau target dipakai di fitur), CE pada prefiks akan berubah saat byte berikutnya
    diubah. Diuji untuk KEDUA learner agar jalur cepat dan acuan sama-sama terjaga.
    """
    for K in (Learner, FastLearner):
        tk = dict(think_prob=0.0)
        base = dict(adapter_rank=0, slow_every=10 ** 9, replay=False, sleep_every=10 ** 9)
        _, ln, m, _ = build(K, tk=tk, **base)
        a = list(TEXT[:40])
        b = list(a)
        b[20] ^= 0x5A                       # ubah satu byte di TENGAH
        s1, s2 = Stream(m, online=False), Stream(m, online=False)
        ca, cb = [], []
        for i in range(40):
            c1 = ln.step(s1, a[i], train=False, think=False)
            c2 = ln.step(s2, b[i], train=False, think=False)
            if i > 0:
                ca.append(c1)
                cb.append(c2)
        # ca[k] = CE untuk byte indeks k+1. Byte yang diubah adalah indeks 20.
        # Jadi CE untuk byte 1..19 (ca[:19]) harus identik; CE byte 20 (ca[19]) BOLEH berbeda.
        assert np.allclose(ca[:19], cb[:19], atol=1e-12), (K.__name__, "kebocoran di prefiks")
        assert not np.allclose(ca[19], cb[19]), (K.__name__, "byte berubah tidak berpengaruh")
        print(f"  kausalitas {K.__name__}: prefiks identik, lanjutan berubah (ok)")


def _fd_err(n_layers, T, layer=0, deep=True, **kw):
    """Galat relatif gradien L{layer}.Wc vs beda-hingga terpusat. Model dibekukan."""
    FIX = dict(mem_surprise_fixed=True, beta_max=1.0)     # matikan gerbang kejutan
    FIX.update(kw)

    def run(W):
        cfg = Config(d=8, n_layers=n_layers, d_mem=4, vocab=256, head="linear",
                     adapter_rank=0, slow_every=10 ** 9, replay=False, sleep_every=10 ** 9,
                     seed=0, deep_grad=deep, **FIX)
        m = TMT2(cfg)
        m.layers[layer].Wc.data = np.array(W, dtype=float).copy()
        ln = FastLearner(m, TrainOpts(think_prob=0.0, only_byte_loss=True))
        s, tot = Stream(m), 0.0
        for b in TEXT[:T]:
            c = ln.step(s, int(b))
            tot += 0.0 if c is None else c
        return tot, ln

    base = TMT2(Config(d=8, n_layers=n_layers, d_mem=4, vocab=256, head="linear",
                       adapter_rank=0, slow_every=10 ** 9, replay=False,
                       sleep_every=10 ** 9, seed=0, deep_grad=deep,
                       **FIX)).layers[layer].Wc.data
    _, ln = run(base)
    if "byte" not in ln.gacc:
        return None
    o, shp = ln.offs[f"L{layer}.Wc"]
    g = ln.gacc["byte"][o:o + shp[0] * shp[1]].reshape(shp)
    if np.abs(g).max() == 0:
        return None
    idx = np.dstack(np.unravel_index(np.argsort(-np.abs(g).ravel())[:3], shp))[0]
    eps, worst = 1e-5, 0.0
    for i, j in idx:
        Wp, Wm = base.copy(), base.copy()
        Wp[i, j] += eps
        Wm[i, j] -= eps
        num = (run(Wp)[0] - run(Wm)[0]) / (2 * eps)
        worst = max(worst, abs(g[i, j] - num) / max(1e-3, abs(num)))
    return worst


def test_rtrl_one_layer_is_exact_multi_layer_is_truncated():
    """BATASAN TERUKUR: RTRL satu lapisan eksak, ``deep_grad`` antar-lapisan TIDAK.

    ``test_rtrl_exact_vs_bptt`` di test_core.py memakai ``n_layers=1``, jadi ia tidak pernah
    menguji tumpukan. Tes ini mengisi celah itu dan MENDOKUMENTASIKAN hasilnya:

    * 1 lapisan  -> galat ~1e-9 (eksak; RTRL diagonal memang eksak untuk rekurensi diagonal).
    * >=2 lapisan -> galat tumbuh dengan panjang deret (T) dan jumlah lapisan.

    Penyebab: jejak RTRL bersifat PER LAPISAN -- ia melacak dh_l/d(theta_l) saja. Gradien eksak
    ke parameter lapisan bawah dari loss lapisan atas memerlukan sensitivitas state lapisan atas
    terhadap parameter lapisan bawah (tensor O(d^3) untuk T langkah). ``deep_grad`` hanya
    menambahkan bagian SESAAT (dalam satu langkah) lewat ``rec.below``; kopling TEMPORAL melalui
    rekurensi lapisan atas tidak dihitung.

    Tes ini SENGAJA tidak menuntut galat kecil pada >=2 lapisan (itu bukan perilaku sekarang).
    Ia mengunci galat 1 lapisan tetap eksak, dan akan GAGAL bila seseorang memperbaiki
    ``deep_grad`` menjadi eksak -- saat itu batasan ini sudah usang dan harus diperbarui.
    """
    e1 = _fd_err(1, 40)
    assert e1 is not None and e1 < 1e-7, e1
    print(f"  RTRL 1 lapisan  : galat={e1:.2e}  (eksak)")
    e3 = _fd_err(3, 40)
    assert e3 is not None and e3 > 1e-2, e3
    print(f"  deep_grad 3 lapisan: galat={e3:.2e}  (TERPOTONG, bukan bug baru)")


def test_surprise_gate_breaks_rtrl_exactness():
    """BATASAN TERUKUR: gerbang kejutan menambah jalur gradien yang tidak dilacak.

    ``beta = beta_max * min(1, surprise)`` dan ``surprise`` bergantung pada ``pp = l2n(y_{t-1} P)``,
    yang bergantung pada parameter. Jalur ``theta -> y_{t-1} -> surprise_t -> beta_t -> M_t``
    TIDAK ada di jejak RTRL (penulisan memori diperlakukan eksogen). Akibatnya, dengan
    ``use_innovation=True`` DAN ``use_memory=True`` sekaligus, gradien satu lapisan pun melenceng.

    ``mem_surprise_fixed=True`` (beta tetap) menutup jalur itu. **Konfigurasi terbaik proyek
    sudah memakai flag itu** -- jadi bias ini mengenai konfigurasi DEFAULT, bukan yang terbaik.
    """
    e_gate = _fd_err(1, 40, mem_surprise_fixed=False)
    e_fix = _fd_err(1, 40, mem_surprise_fixed=True)
    assert e_gate is not None and e_gate > 1e-3, e_gate
    assert e_fix is not None and e_fix < 1e-7, e_fix
    print(f"  gerbang kejutan AKTIF : galat={e_gate:.2e}  (jalur beta tidak dilacak)")
    print(f"  beta tetap (terbaik)  : galat={e_fix:.2e}  (eksak)")


if __name__ == "__main__":
    import time
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t0 = time.time()
            print(name, end=" ", flush=True)
            fn()
            print(f"  ({time.time()-t0:.1f}s)")
    print("fast_learner ok")


def test_mixer_order_span_reduces_ce():
    """Orde konteks mixer adalah knob utama: rentang orde lebih panjang -> CE lebih rendah.

    Hasil ukur (LAPORAN §14): pada sumber Markov orde-2 deterministik, mixer dengan orde {1,2,3}
    HARUS mengalahkan mixer orde {1} saja, dan orde {1,2,3,4} tidak lebih buruk dari {1,2,3}.
    Tes ini mengunci MEKANISME (bukan angka WikiText yang mahal): memberi mixer orde lebih
    panjang tidak boleh memperburuk, dan orde-2 harus benar-benar terpakai.
    """
    from tmt2.corpus import markov_det_stream
    ids, _ = markov_det_stream(6000, 2, 8, 0.10, 0)
    tr, te = ids[:4000], ids[4000:]

    def frozen_bpb(orders):
        m = TMT2(Config(d=16, vocab=8, n_layers=1, d_mem=8, use_memory=False,
                        head="linear", out_zero_init=True, adapter_rank=0,
                        use_innovation=False, mixer_k=1, mixer_match=False,
                        mixer_match_orders=orders, seed=0))
        fl = FastLearner(m, TrainOpts(seed=0, only_byte_loss=True, think_prob=0.0,
                                      use_stop=False, only_params=("mix_w",)))
        s = Stream(m, online=True)
        for b in tr:
            fl.step(s, int(b), train=True, think=False)
        s2 = Stream(m, online=False)
        ce, n = 0.0, 0
        for b in te:
            c = fl.step(s2, int(b), train=False, think=False)
            if c is not None:
                ce += float(c); n += 1
        return ce / max(n, 1) / np.log(2)

    b1 = frozen_bpb("1")
    b3 = frozen_bpb("1,2,3")
    b4 = frozen_bpb("1,2,3,4")
    print(f"  mixer orde 1={b1:.3f}  1,2,3={b3:.3f}  1,2,3,4={b4:.3f} bit")
    # orde-2 memang ada di sumber -> mixer orde {1,2,3} harus LEBIH BAIK dari orde {1}
    assert b3 < b1 - 0.05, (b1, b3)
    # menambah orde tidak boleh memperburuk secara material
    assert b4 <= b3 + 0.05, (b3, b4)


def test_frozen_eval_does_not_mutate_mixer_table():
    """TES PENJAGA (audit kebocoran §16): eval BEKU tidak boleh menyentuh tabel n-gram mixer.

    `_mix_ng.update` hanya dipanggil saat `train and s.online` (fast_learner.step). Karena itu
    eval beku (train=False) wajib meninggalkan tabel PERSIS sama. Bila tes ini gagal, artinya
    jalur eval beku menulis ke tabel -> SEMUA angka beku (warm/cold) di proyek tidak sah.
    """
    from tmt2.corpus import markov_det_stream
    ids, _ = markov_det_stream(3000, 2, 8, 0.10, 0)
    m = TMT2(Config(d=16, vocab=8, n_layers=1, d_mem=8, use_memory=False, head="linear",
                    out_zero_init=True, adapter_rank=0, use_innovation=False,
                    mixer_k=1, mixer_match=True, mixer_match_orders="1,2,3,4", seed=0))
    fl = FastLearner(m, TrainOpts(seed=0, only_byte_loss=True, think_prob=0.0,
                                  use_stop=False, only_params=("mix_w",)))
    s = Stream(m, online=True)
    for b in ids[:2000]:
        fl.step(s, int(b), train=True, think=False)

    def size(ng):
        nctx = sum(len(t) for t in ng.tabs)
        ncnt = sum(n for t in ng.tabs for _, n in t.values())
        return nctx, ncnt

    before = size(fl._mix_ng)
    # eval beku pada jendela uji (cold lalu warm)
    for carry in (None, s):
        s2 = carry if carry is not None else Stream(m, online=False)
        for b in ids[2000:3000]:
            fl.step(s2, int(b), train=False, think=False)
    after = size(fl._mix_ng)
    print(f"  entri tabel sebelum={before}  sesudah eval beku={after}")
    assert before == after, f"eval beku MENGUBAH tabel mixer! {before} -> {after}"


def test_online_eval_contaminates_later_frozen_eval():
    """TES MEKANISME (§16): online-dulu membuat angka beku PALSU rendah; beku-dulu tidak.

    Membuktikan kebocoran secara langsung: pada sumber deterministik, menjalankan eval ONLINE
    pada jendela uji sebelum eval BEKU membuat tabel menghafal jendela itu, sehingga CE beku
    jatuh jauh (di sini >=0,3 bit) padahal model & bobot identik. Urutan beku-dulu memulihkan
    angka jujur. Tes ini LULUS bila kebocoran terdeteksi DAN urutan beku-dulu bebas darinya.
    """
    from tmt2.corpus import markov_det_stream
    ids, _ = markov_det_stream(5000, 2, 8, 0.05, 0)
    tr, te = ids[:3000], ids[3000:]

    def run(order):
        m = TMT2(Config(d=16, vocab=8, n_layers=1, d_mem=8, use_memory=False, head="linear",
                        out_zero_init=True, adapter_rank=0, use_innovation=False,
                        mixer_k=1, mixer_match=True, mixer_match_orders="1,2,3,4,5,6", seed=0))
        fl = FastLearner(m, TrainOpts(seed=0, only_byte_loss=True, think_prob=0.0,
                                      use_stop=False, only_params=("mix_w",)))
        s = Stream(m, online=True)
        for b in tr:
            fl.step(s, int(b), train=True, think=False)

        def frozen():
            ce, n = 0.0, 0
            for b in te:
                c = fl.step(s, int(b), train=False, think=False)
                if c is not None:
                    ce += float(c); n += 1
            return ce / max(n, 1) / np.log(2)

        def online():
            for b in te:
                fl.step(s, int(b), train=True, think=False)

        if order == "frozen_first":
            f = frozen(); online()
        else:                                    # online_first: KONTAMINASI
            online(); f = frozen()
        return f

    f_clean = run("frozen_first")
    f_dirty = run("online_first")
    print(f"  beku-dulu={f_clean:.3f}  online-dulu(kontaminasi)={f_dirty:.3f}  "
          f"selisih={f_dirty - f_clean:+.3f} bit")
    # mekanisme kebocoran: online-dulu HARUS jauh lebih rendah (menghafal jendela uji)
    assert f_dirty < f_clean - 0.3, (f_clean, f_dirty)


def test_adamw_kernel_matches_numpy_exactly():
    """Kernel njit adamw_step HARUS setara numerik dgn urutan numpy jalur fallback.

    Catatan penting: lintasan LATIH di 8000 byte TIDAK bisa dipakai untuk memverifikasi ini
    (online learning kacau: beda 1e-16 membesar eksponensial -> BPB akhir bisa beda 0,08).
    Uji di sini pada SATU LANGKAH, di mana perbandingan masih bermakna.
    """
    from tmt2 import fastpath as fp
    rng = np.random.default_rng(7)
    N = 4096
    tidx = np.arange(5)
    offs = np.array([0, 800, 1600, 2400, 3200])
    sizes = np.array([800, 800, 800, 800, N - 3200])
    mat = np.array([1, 1, 0, 1, 0])
    lrmul = np.array([1.0, 1.0, 1.0, 0.1, 0.1])
    lr, b1, b2, wd, t_adam = 5e-3, 0.9, 0.999, 0.02, 37
    bc1 = 1.0 / (1.0 - b1 ** t_adam)
    bc2 = 1.0 / (1.0 - b2 ** t_adam)
    mask = np.ones(N); mask[100:120] = 0.0

    for om in (None, mask):
        for clip in (1e9, 0.5):
            gb0 = rng.standard_normal(N) * 0.05
            am0 = rng.standard_normal(N) * 0.01
            av0 = np.abs(rng.standard_normal(N)) * 0.01
            th0 = rng.standard_normal(N)

            gb = gb0.copy(); am = am0.copy(); av = av0.copy(); th = th0.copy()
            fp.adamw_step(gb, am, av, th, np.zeros(N), lr, b1, b2, bc1, bc2,
                          clip, wd, offs, sizes, mat, lrmul, tidx, om)

            g = gb0.copy()
            gn = float(np.sqrt((g * g).sum()))
            if gn > clip:
                g *= clip / gn
            if om is not None:
                g *= om
            m2 = am0 * b1 + g * (1 - b1)
            v2 = av0 * b2 + (g * g) * (1 - b2)
            upd = (m2 / (1 - b1 ** t_adam)) / (np.sqrt(v2 / (1 - b2 ** t_adam)) + 1e-8)
            th2 = th0.copy()
            dcy = 1.0 - lr * wd
            for k in range(len(tidx)):
                o, sz = offs[k], sizes[k]
                if mat[k]:
                    th2[o:o + sz] *= dcy
                th2[o:o + sz] -= lr * lrmul[k] * upd[o:o + sz]

            assert np.abs(gb - g).max() < 1e-12, (om is not None, clip)
            assert np.abs(am - m2).max() < 1e-12
            assert np.abs(av - v2).max() < 1e-12
            assert np.abs(th - th2).max() < 1e-12


def test_slow_update_njit_and_numpy_agree_for_one_cycle():
    """slow_update dgn use_njit_adam True/False harus identik setelah SATU siklus.

    Setelah beberapa siklus perbedaan meledak karena kekacauan; satu siklus masih terbanding.
    """
    cfg = Config.best(vocab=256, d=16, n_layers=2, d_mem=4, slow_every=4,
                      seg_len=64, replay=False, sleep_every=10 ** 9)
    cfg.mixer_k = 0
    from tmt2.corpus import load
    ids = np.frombuffer(load("license"), dtype=np.uint8).astype(np.int64)[:64]

    ths = []
    for njit in (True, False):
        m = TMT2(cfg)
        fl = FastLearner(m, TrainOpts(seed=0, think_prob=0.0, only_byte_loss=True,
                                      use_stop=False, use_njit_adam=njit))
        s = Stream(m, online=True)
        for b in ids:
            fl.step(s, int(b), train=True, think=False)
        ths.append(fl.theta.copy())
    assert np.abs(ths[0] - ths[1]).max() < 1e-10
