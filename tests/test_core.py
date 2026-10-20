"""Uji inti: RTRL diagonal harus EKSAK terhadap BPTT pada graf yang di-unroll."""
import numpy as np
from tmt2.autograd import *
from tmt2.autograd import CTX
from tmt2.model import Config, TMT2
from tmt2.core import (LayerState, StackState, layer_step, rtrl_flush, backward_task,
                       ff_readout, stack_step, byte_logits)

LAYER_PARAMS = ["Wa", "ba", "B", "K", "Wc", "Wg1", "Wg2", "Wo", "Af", "Bf", "Wq", "Wm"]


def small_setup(seed=0, T=14, **kw):
    rng = np.random.default_rng(seed)
    cfg = Config(d=6, n_layers=1, expand=2, d_mem=4, adapter_rank=2, horizons=(1,), **kw)
    model = TMT2(cfg)
    L = model.layers[0]
    for n in LAYER_PARAMS:                       # acak SEMUA (K=0 & Bf=0 bawaan membuat tes kosong)
        p = getattr(L, n)
        p.data = rng.normal(0, 0.6, p.data.shape) + (0.5 if n == "ba" else 0.0)
    xs = rng.normal(size=(T, cfg.d))
    es = 0.5 * rng.normal(size=(T, cfg.d))
    sur = rng.uniform(0, 1.5, T)
    U = rng.normal(size=(T, cfg.d))
    w = rng.uniform(0.5, 1.5, T)
    return cfg, model, L, xs, es, sur, U, w


def step_loss(y, U, w, t):
    return tsum((y - U[t]) * (y - U[t])) * w[t]


def ground_truth(cfg, L, xs, es, Ms, U, w):
    """BPTT penuh: rekurensi di-unroll dengan autograd; x, e, M eksogen."""
    for n in LAYER_PARAMS:
        getattr(L, n).tg.pop("gt", None)
    CTX.task = "gt"
    h, total = Tensor(np.zeros(cfg.d)), 0.0
    for t in range(len(xs)):
        x = Tensor(xs[t])
        a = sigmoid(matmul(x, L.Wa) + L.ba)
        h = a * h + (1.0 - a) * matmul(x, L.B)
        if cfg.use_innovation:
            h = h + matmul(Tensor(es[t]), L.K)
        total = total + step_loss(ff_readout(cfg, L, x, h, Ms[t]), U, w, t)
    total.backward()
    CTX.task = "default"


def rtrl_run(cfg, L, xs, es, sur):
    st, recs, Ms = LayerState(cfg.d, cfg.d_mem), [], []
    for t in range(len(xs)):
        Ms.append(st.M)
        rec, st = layer_step(cfg, L, Tensor(xs[t]), st, es[t], sur[t])
        recs.append(rec)
    return recs, Ms


def compare(L, tag_a, tag_b, tol=1e-9):
    worst = 0.0
    for n in LAYER_PARAMS:
        p = getattr(L, n)
        ga, gb = p.tg[tag_a], p.tg[tag_b]
        err = np.abs(ga - gb).max() / (np.abs(ga).max() + 1e-12)
        assert np.abs(ga).max() > 1e-8, f"{n}: gradien nol -> tes kosong"
        assert err < tol, (n, err)
        worst = max(worst, err)
    return worst


def test_rtrl_exact_vs_bptt():
    cfg, model, L, xs, es, sur, U, w = small_setup()
    recs, Ms = rtrl_run(cfg, L, xs, es, sur)
    ground_truth(cfg, L, xs, es, Ms, U, w)
    for n in LAYER_PARAMS:
        getattr(L, n).tg.pop("r", None)
    for t, rec in enumerate(recs):               # loss tiap langkah, diselesaikan segera
        backward_task(step_loss(rec.y, U, w, t), [rec], "r")
    worst = compare(L, "gt", "r")
    print(f"  RTRL vs BPTT (segera): galat relatif maks = {worst:.2e}")


def test_rtrl_delayed_resolution():
    """Loss diselesaikan TERLAMBAT dan tak berurutan (seperti horizon 4/16): jejak S_t
    harus disimpan per langkah, hasilnya identik."""
    cfg, model, L, xs, es, sur, U, w = small_setup(seed=1)
    recs, Ms = rtrl_run(cfg, L, xs, es, sur)
    ground_truth(cfg, L, xs, es, Ms, U, w)
    for n in LAYER_PARAMS:
        getattr(L, n).tg.pop("r", None)
    for t in reversed(range(len(recs))):         # setelah SELURUH deret, urutan terbalik
        backward_task(step_loss(recs[t].y, U, w, t), [recs[t]], "r")
    worst = compare(L, "gt", "r")
    print(f"  RTRL vs BPTT (terlambat, terbalik): galat relatif maks = {worst:.2e}")


def test_tasks_are_separate():
    """Dua task pada rec yang sama: gradien per-task terpisah (bahan PCGrad) dan
    jumlahnya = gradien gabungan."""
    cfg, model, L, xs, es, sur, U, w = small_setup(seed=2, T=8)
    recs, Ms = rtrl_run(cfg, L, xs, es, sur)
    ground_truth(cfg, L, xs, es, Ms, U, w)
    for n in LAYER_PARAMS:
        p = getattr(L, n)
        for k in ("r1", "r2"):
            p.tg.pop(k, None)
    for t, rec in enumerate(recs):
        backward_task(step_loss(rec.y, U, w, t) * 0.25, [rec], "r1")
        backward_task(step_loss(rec.y, U, w, t) * 0.75, [rec], "r2")
    for n in LAYER_PARAMS:
        p = getattr(L, n)
        s = p.tg["r1"] + p.tg["r2"]
        assert np.allclose(s, p.tg["gt"], rtol=1e-8, atol=1e-10), n
        assert not np.allclose(p.tg["r1"], p.tg["r2"])
    print("  gradien per-task terpisah & jumlahnya cocok")


def test_no_innovation_no_memory_variants():
    for kw in (dict(use_innovation=False), dict(use_memory=False)):
        cfg, model, L, xs, es, sur, U, w = small_setup(seed=3, T=8, **kw)
        recs, Ms = rtrl_run(cfg, L, xs, es, sur)
        global LAYER_PARAMS
        keep = LAYER_PARAMS
        LAYER_PARAMS = [n for n in keep
                        if not (kw.get("use_innovation") is False and n == "K")
                        and not (kw.get("use_memory") is False and n in ("Wq", "Wm"))]
        try:
            ground_truth(cfg, L, xs, es, Ms, U, w)
            for n in LAYER_PARAMS:
                getattr(L, n).tg.pop("r", None)
            for t, rec in enumerate(recs):
                backward_task(step_loss(rec.y, U, w, t), [rec], "r")
            worst = compare(L, "gt", "r")
        finally:
            LAYER_PARAMS = keep
        print(f"  varian {kw}: galat = {worst:.2e}")


def test_stack_stream_smoke():
    """Tumpukan penuh mengalir pada byte nyata: hingga, terbatas, CE awal wajar,
    dan stop-grad antar lapisan benar (lapisan bawah tak dapat gradien CE).

    Catatan `head`: dengan `head="linear"` (default sejak 2026-10-01) kamus keluaran
    adalah `W_out`, BUKAN encoder `E`. Jadi gradien CE dari lapisan puncak tidak lagi
    menyentuh `E` langsung -- `E` hanya belajar dari loss JEPA per-lapisan. Uji ini
    karena itu memakai `head="cosine"` secara eksplisit untuk memeriksa jalur lama
    (CE -> E), lalu mengulang tanpa asumsi itu untuk default baru.
    """
    for head in ("cosine", "linear"):
        cfg = Config(d=16, n_layers=3, d_mem=8, head=head)
        model = TMT2(cfg)
        text = (b"the quick brown fox jumps over the lazy dog. " * 6)[:200]
        tgt = model.target_dict()
        st, prev_logits, ces = StackState(model), None, []
        for t, b in enumerate(text):
            if prev_logits is not None:
                ces.append(float(cross_entropy(prev_logits, b).data))
            x = model.encode_byte(b)
            recs, st = stack_step(model, x, st, tgt[b])
            prev_logits = byte_logits(model, recs[-1].pred(model, cfg.n_layers - 1, 1)).data
            for s in st.layers:
                assert np.isfinite(s.h).all() and np.abs(s.h).max() < 50
        print(f"  [{head}] CE awal rata-rata = {np.mean(ces):.3f} (ln 256 = {np.log(256):.3f})")
        assert np.isfinite(ces).all() and np.mean(ces) < 9.0
        # stop-grad: CE dari lapisan puncak tidak boleh menyentuh lapisan bawah
        for p in model.params.values():
            p.tg.clear()
        logits = byte_logits(model, recs[-1].pred(model, cfg.n_layers - 1, 1))
        backward_task(cross_entropy(logits, 65), [recs[-1]], "ce")
        top, low = model.layers[-1], model.layers[0]
        assert "ce" in top.Wa.tg and "ce" in top.Wc.tg
        # kepala kosinus memakai E sebagai kamus keluaran -> E dapat gradien CE.
        # kepala linear memakai W_out -> E TIDAK dapat gradien CE (hanya dari JEPA).
        out_name = model.out_dict_name()
        assert "ce" in getattr(model, out_name).tg, (head, out_name)
        if head == "cosine":
            assert out_name == "E" and "ce" in model.E.tg
        assert "ce" not in low.Wa.tg and "ce" not in low.Wc.tg
        print(f"  [{head}] stop-grad antar lapisan: OK; kamus keluaran = {out_name}")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name); fn()
    print("core ok")
