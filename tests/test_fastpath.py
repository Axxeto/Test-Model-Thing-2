"""Validasi jalur panas @njit terhadap autograd (definisi kebenaran).

Setiap kernel di tmt2/fastpath.py dibandingkan dengan padanannya di tmt2/core.py +
tmt2/autograd.py pada nilai acak. Toleransi ~1e-12 (float64, urutan operasi berbeda).

  PYTHONPATH=. python tests/test_fastpath.py
"""
import numpy as np
from tmt2.autograd import Tensor, CTX, cross_entropy, l2norm, matmul, tsum
from tmt2.model import Config, TMT2, ln_np, l2n_np
from tmt2.core import LayerState, ff_readout, layer_step, rtrl_flush as rtrl_flush_ref, byte_logits
from tmt2 import fastpath as fp

LAYER_PARAMS = ["Wa", "ba", "B", "K", "Wc", "Wg1", "Wg2", "Wo", "Af", "Bf", "Wq", "Wm"]


def setup(seed=0, **kw):
    rng = np.random.default_rng(seed)
    o = dict(d=6, n_layers=1, expand=2, d_mem=4, adapter_rank=2, horizons=(1,))
    o.update(kw)
    cfg = Config(**o)
    model = TMT2(cfg)
    L = model.layers[0]
    for n in LAYER_PARAMS:
        p = getattr(L, n)
        p.data = rng.normal(0, 0.6, p.data.shape) + (0.5 if n == "ba" else 0.0)
    d, dm = cfg.d, cfg.d_mem
    h = rng.normal(size=d)
    x = rng.normal(size=d)
    M = rng.normal(size=(dm, dm)) * 0.3
    return cfg, model, L, rng, h, x, M


def relerr(a, b):
    return float(np.abs(a - b).max() / (np.abs(b).max() + 1e-12))


def test_readout_fwd():
    for kw in ({}, dict(use_memory=False), dict(adapter_rank=0), dict(use_memory=False, adapter_rank=0)):
        cfg, model, L, rng, h, x, M = setup(**kw)
        ref = ff_readout(cfg, L, Tensor(x), Tensor(h, requires_grad=True), M).data
        fast = fp.readout_fwd(h, x, M, L.Wc.data, L.Wq.data, L.Wm.data, L.Wg1.data,
                              L.Wg2.data, L.Wo.data, L.Af.data, L.Bf.data,
                              cfg.use_memory, cfg.adapter_rank > 0, cfg.mem_query_x)
        e = relerr(fast, ref)
        assert e < 1e-12, (kw, e)
        print(f"  readout_fwd {kw or 'default'}: galat={e:.2e}")


def test_readout_bwd():
    for kw in ({}, dict(use_memory=False), dict(adapter_rank=0)):
        cfg, model, L, rng, h, x, M = setup(seed=1, **kw)
        gy = rng.normal(size=cfg.d)
        # -- referensi: backward autograd dengan seed gy
        for p in L.list:
            p.tg.pop("t", None)
        h_leaf = Tensor(h, requires_grad=True)
        y = ff_readout(cfg, L, Tensor(x), h_leaf, M)
        CTX.task = "t"
        y.backward(seed=gy)
        CTX.task = "default"
        g_h_ref, g_x_ref = h_leaf.grad, gy
        # -- jalur panas
        g_h, g_x, gWc, gWq, gWm, gWg1, gWg2, gWo, gAf, gBf = fp.readout_bwd(
            h, x, M, gy, L.Wc.data, L.Wq.data, L.Wm.data, L.Wg1.data, L.Wg2.data,
            L.Wo.data, L.Af.data, L.Bf.data, cfg.use_memory, cfg.adapter_rank > 0,
            cfg.mem_query_x)
        assert relerr(g_h, g_h_ref) < 1e-12
        assert relerr(g_x, g_x_ref) < 1e-12
        for name, got in (("Wc", gWc), ("Wq", gWq), ("Wm", gWm), ("Wg1", gWg1), ("Wg2", gWg2),
                          ("Wo", gWo), ("Af", gAf), ("Bf", gBf)):
            ref = getattr(L, name).tg.get("t")
            if ref is None:
                assert not np.any(got), (kw, name)
                continue
            e = relerr(got, ref)
            assert e < 1e-12, (kw, name, e)
        print(f"  readout_bwd {kw or 'default'}: OK")


def test_rec_fwd_and_rtrl():
    for use_innov in (True, False):
        cfg, model, L, rng, h, x, M = setup(seed=2, use_innovation=use_innov)
        st = LayerState(cfg.d, cfg.d_mem)
        st.h = h
        st.S = {k: rng.normal(size=v.shape) for k, v in st.S.items()}
        e = rng.normal(size=cfg.d) * 0.5
        rec, ns = layer_step(cfg, L, Tensor(x), st, e, 0.7)
        use_e = use_innov and e is not None
        h_f, SWa, SB, SK, Sba = fp.rec_fwd(x, h, st.S["Wa"], st.S["B"], st.S["K"], st.S["ba"],
                                           L.Wa.data, L.ba.data, L.B.data, L.K.data, e, use_e)
        assert relerr(h_f, ns.h) < 1e-12
        for nm, got in (("Wa", SWa), ("B", SB), ("K", SK), ("ba", Sba)):
            assert relerr(got, ns.S[nm]) < 1e-12, (use_innov, nm)
        # rtrl_flush memakai jejak S_t (state BARU), sama seperti core.rtrl_flush
        g = rng.normal(size=cfg.d)
        gWa, gB, gba, gK = fp.rtrl_flush(g, ns.S["Wa"], ns.S["B"], ns.S["K"], ns.S["ba"], use_e)
        for p in L.list:
            p.tg.pop("z", None)
        rec.h_leaf.grad = g.copy()
        rtrl_flush_ref(rec, "z")
        for nm, got in (("Wa", gWa), ("B", gB), ("K", gK), ("ba", gba)):
            ref = getattr(L, nm).tg.get("z")
            if ref is None:
                assert not np.any(got), (use_innov, nm)
                continue
            assert relerr(got, ref) < 1e-12, (use_innov, nm)
        print(f"  rec_fwd+rtrl_flush use_innovation={use_innov}: OK")


def test_head():
    # Uji kernel kepala KOSINUS (head_fwd/head_bwd). Kepala linear (default sekarang)
    # diuji di test_head_linear di bawah. Config default kini "linear" -> eksplisit di sini.
    cfg = Config(d=6, n_layers=1, d_mem=4, adapter_rank=2, head="cosine")
    model = TMT2(cfg)
    rng = np.random.default_rng(5)
    y = rng.normal(size=cfg.d)
    P = rng.normal(size=(cfg.d, cfg.d)) * 0.3
    D = l2norm(Tensor(model.E.data)).data
    pred_t = matmul(Tensor(y, requires_grad=True), Tensor(P, requires_grad=True))
    logits_t = byte_logits(model, pred_t)                 # D=None -> graf l2norm(E) ikut terbangun
    fast = fp.head_fwd(y @ P, D, float(model.logit_scale.data), model.byte_bias.data)
    assert relerr(fast, logits_t.data) < 1e-12
    b = 65
    for k in ("E", "byte_bias", "logit_scale"):
        model.params[k].tg.pop("t", None)
    CTX.task = "t"
    cross_entropy(logits_t, b).backward()
    CTX.task = "default"
    glo = fp.softmax_grad_target(fast, b, 1.0)
    g_y, gP, gE, g_ls, g_bb = fp.head_bwd(y @ P, D, model.E.data,
                                          float(model.logit_scale.data), glo, 1.0, P, y)
    assert relerr(g_bb, model.byte_bias.tg["t"]) < 1e-12
    assert relerr(np.array(g_ls), model.logit_scale.tg["t"]) < 1e-12
    assert relerr(gE, model.E.tg["t"]) < 1e-12
    assert relerr(g_y, pred_t._parents[0].grad) < 1e-12
    assert relerr(gP, pred_t._parents[1].grad) < 1e-12
    print("  head_fwd/head_bwd (g_y, gP, gE, logit_scale, byte_bias): OK")


def test_head_linear():
    """Kepala LINEAR (default sejak 2026-10-01): logits = W_out @ l2norm(p) + byte_bias.

    Diuji terhadap autograd `byte_logits` untuk fwd, dan terhadap backward CE untuk
    (g_y, gP, g_W_out, g_byte_bias)."""
    cfg = Config(d=6, n_layers=1, d_mem=4, adapter_rank=2, head="linear")
    model = TMT2(cfg)
    rng = np.random.default_rng(11)
    y = rng.normal(size=cfg.d)
    P = rng.normal(size=(cfg.d, cfg.d)) * 0.3
    Wo = model.W_out.data
    pred_t = matmul(Tensor(y, requires_grad=True), Tensor(P, requires_grad=True))
    logits_t = byte_logits(model, pred_t)
    fast = fp.head_fwd_lin(y @ P, Wo, model.byte_bias.data)
    assert relerr(fast, logits_t.data) < 1e-12
    b = 65
    for k in ("W_out", "byte_bias"):
        model.params[k].tg.pop("t", None)
    CTX.task = "t"
    cross_entropy(logits_t, b).backward()
    CTX.task = "default"
    glo = fp.softmax_grad_target(fast, b, 1.0)
    g_y, gP, gWo, g_bb = fp.head_bwd_lin(y @ P, Wo, glo, 1.0, P, y)
    assert relerr(g_bb, model.byte_bias.tg["t"]) < 1e-12
    assert relerr(gWo, model.W_out.tg["t"]) < 1e-12
    assert relerr(g_y, pred_t._parents[0].grad) < 1e-12
    assert relerr(gP, pred_t._parents[1].grad) < 1e-12
    print("  head_fwd_lin/head_bwd_lin (g_y, gP, g_W_out, byte_bias): OK")


def test_fused_multibyte_equivalence():
    """KERNEL TERPADU vs jalur lama harus BIT-IDENTIK lintas byte, dengan memori ON.

    Regresi untuk dua bug halus di `step_fused`:
      (a) slot memori arena: readout/backward butuh M SEBELUM tulis (Mpre), state yang
          mengalir butuh M SESUDAH tulis (Mpost) -- satu slot saja salah.
      (b) URUTAN snapshot: `f_st_prev` harus disalin SEBELUM `fused_promote_mem` (promote
          menimpa Mpre), jika tidak backward byte berikutnya membaca memori yang salah.
    Gejala bila salah: h/a/cc identik, M beda, gradien semua blok menyimpang mulai byte 2.
    """
    from tmt2.learner import Stream, TrainOpts
    from tmt2.fast_learner import FastLearner

    def build(fused):
        cfg = Config.best(d=12, n_layers=2, vocab=48, d_mem=6, use_memory=True,
                          head="linear", deep_grad=True, adapter_rank=0, mixer_k=0,
                          think_max=0, use_innovation=False)
        m = TMT2(cfg)
        o = TrainOpts(only_byte_loss=True, use_fused_kernel=fused, seed=0, use_stop=False)
        return m, FastLearner(m, o)

    data = [3, 17, 42, 5, 29]
    mR, LA = build(False)
    mF, LB = build(True)
    sA, sB = Stream(mR, online=True), Stream(mF, online=True)
    for i, b in enumerate(data):
        LA.step(sA, b, train=True)
        LB.step_fused(sB, b, train=True)
        gA, gB = LA.gacc.get("byte"), LB.gacc.get("byte")
        if gA is None or gB is None:
            continue
        dmax = float(np.abs(gA - gB).max())
        assert dmax < 1e-12, f"byte {i}: gradien arena beda {dmax:.3e}"
        # CE prekuensial harus sama juga
        assert abs(float(sA.seg_ce) - float(sB.seg_ce)) < 1e-12
    print("  step_fused (memori ON, 5 byte): gradien arena + CE bit-identik: OK")


def test_fused_committed_params_match():
    """Setelah BANYAK byte (beberapa `slow_update`), parameter harus tetap sama.

    Regresi untuk bug ketiga: `step_fused` dulu tidak memanggil `_note("byte", ce)` sehingga
    pembaruan ketidakpastian Kendall (`unc["byte"]`, bobot `exp(-unc)`) tidak pernah berjalan.
    Akibatnya `gb = gacc/K * exp(-unc)` berbeda ~5% setelah slow_update pertama.
    """
    from tmt2.learner import Stream, TrainOpts
    from tmt2.fast_learner import FastLearner

    def build(fused):
        cfg = Config.best(d=10, n_layers=2, vocab=40, d_mem=4, use_memory=True,
                          head="linear", deep_grad=True, adapter_rank=0, mixer_k=0,
                          think_max=0, use_innovation=False, slow_every=4)
        o = TrainOpts(only_byte_loss=True, use_fused_kernel=fused, seed=0,
                      use_stop=False)
        m = TMT2(cfg)
        return m, FastLearner(m, o)

    data = [(i * 7 + 3) % 40 for i in range(60)]      # 14 slow_update
    mR, LA = build(False)
    mF, LB = build(True)
    sA, sB = Stream(mR, online=True), Stream(mF, online=True)
    worst_g = 0.0
    worst_ce = 0.0
    for b in data:
        ca = LA.step(sA, b, train=True)
        cb = LB.step_fused(sB, b, train=True)
        if ca is not None:
            worst_ce = max(worst_ce, abs(float(ca) - float(cb)))
        gA, gB = LA.gacc.get("byte"), LB.gacc.get("byte")
        if gA is not None and gB is not None:
            worst_g = max(worst_g, float(np.abs(gA - gB).max()))
    assert worst_g < 1e-12, f"gacc beda {worst_g:.3e}"
    assert worst_ce < 1e-12, f"CE per-byte beda {worst_ce:.3e}"
    # bandingkan theta lengkap (arena datar)
    dmax = 0.0
    for k, (off, shp) in LA.offs.items():
        n = int(np.prod(shp)) if len(shp) else 1
        dmax = max(dmax, float(np.abs(LA.theta[off:off + n] - LB.theta[off:off + n]).max()))
    assert dmax < 1e-10, f"theta beda setelah 14 slow_update: {dmax:.3e}"
    print(f"  60 byte/14 slow_update: gacc {worst_g:.2e} CE {worst_ce:.2e} theta {dmax:.2e}: OK")


def test_jepa():
    cfg = Config(d=6, n_layers=1, d_mem=4, adapter_rank=2)
    rng = np.random.default_rng(7)
    y = rng.normal(size=cfg.d)
    P = rng.normal(size=(cfg.d, cfg.d)) * 0.3
    tgt = l2n_np(rng.normal(size=cfg.d))
    pred_t = matmul(Tensor(y, requires_grad=True), Tensor(P, requires_grad=True))
    loss_t = 0.5 * tsum((l2norm(pred_t) - tgt) ** 2)
    loss_t.backward()
    lf, g_y, gP = fp.jepa_bwd(y @ P, tgt, P, y)
    assert relerr(np.array(lf), np.array(float(loss_t.data))) < 1e-12
    assert relerr(g_y, pred_t._parents[0].grad) < 1e-12
    assert relerr(gP, pred_t._parents[1].grad) < 1e-12
    print("  jepa_bwd: OK")


def test_stop():
    cfg = Config(d=6, n_layers=1, d_mem=4)
    rng = np.random.default_rng(9)
    n = 3
    sp = rng.uniform(0.05, 0.95, n)
    ces = [3.0, 2.0, 2.5]
    g = fp.stop_grads(sp, np.array(ces), cfg.think_cost, cfg.stop_kl, cfg.stop_prior)
    # referensi numerik (beda-hingga pada stop_loss)
    num = np.zeros(n)
    eps = 1e-6
    for i in range(n):
        a, b = sp.copy(), sp.copy()
        a[i] += eps
        b[i] -= eps
        num[i] = (fp.stop_loss(a, np.array(ces), cfg.think_cost, cfg.stop_kl, cfg.stop_prior)
                  - fp.stop_loss(b, np.array(ces), cfg.think_cost, cfg.stop_kl, cfg.stop_prior)) / (2 * eps)
    e = relerr(g, num)
    assert e < 1e-5, e
    print(f"  stop_grads vs beda-hingga: galat={e:.2e}")


if __name__ == "__main__":
    import time
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t0 = time.time()
            print(name, end=" ", flush=True)
            fn()
            print(f"  ({time.time()-t0:.1f}s)")
    print("fastpath ok")
