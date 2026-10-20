"""Uji learner: komponen kecil (halting, PCGrad, target span), invarian (eval tanpa belajar,
tidur tidak menyentuh jalur cepat, checkpoint bulat-balik, determinisme), dan satu uji belajar."""
import os, tempfile
import numpy as np
from tmt2.autograd import Tensor, cross_entropy
from tmt2.model import Config, TMT2, l2n_np
from tmt2.core import StackState
from tmt2.learner import Learner, Stream, TrainOpts, halting, pcgrad, ce_np, W_TASKS

TEXT = (b"the quick brown fox jumps over the lazy dog. " * 200)   # 9000 byte


def mk(seed=0, tk=None, **kw):
    o = dict(d=16, n_layers=3, d_mem=8, slow_every=8, seg_len=16, sleep_every=64,
             sleep_segments=1, lr=3e-2, seed=seed)
    o.update(kw)
    m = TMT2(Config(**o))
    m.logit_scale.data = np.array(np.log(3.0))
    return Learner(m, TrainOpts(seed=seed, **(tk or {}))), m


def feed(ln, m, data, s=None, **kw):
    s = s or Stream(m)
    ces = []
    for b in data:
        c = ln.step(s, b, **kw)
        if c is not None:
            ces.append(c)
    return s, np.array(ces)


def snap(m):
    return {k: p.data.copy() for k, p in m.params.items()}


def same(a, b):
    return all(np.array_equal(a[k], b[k]) for k in a)


# ------------------------------------------------------------------ komponen kecil
def test_halting():
    h = halting([0.3, 0.5, 0.9])
    assert np.allclose(h, [0.3, 0.35, 0.35]) and abs(h.sum() - 1) < 1e-12
    assert np.allclose(halting([0.7]), [1.0])            # satu kedalaman menyerap semua
    assert abs(halting(np.random.default_rng(0).uniform(size=5)).sum() - 1) < 1e-12


def test_pcgrad():
    rng = np.random.default_rng(0)
    g1, g2 = np.array([1.0, 0.0]), np.array([-1.0, 1.0])          # bertentangan (dot = -1)
    out = pcgrad({"a": g1, "b": g2}, rng)
    assert out["a"] @ g2 >= -1e-9 and out["b"] @ g1 >= -1e-9   # eps 1e-12 di penyebut menyisakan residu
    assert np.array_equal(g1, [1.0, 0.0]) and np.array_equal(g2, [-1.0, 1.0])   # masukan utuh
    ok = pcgrad({"a": np.array([1.0, 0.0]), "b": np.array([1.0, 1.0])}, rng)      # tak bertentangan
    assert np.allclose(ok["a"], [1.0, 0.0]) and np.allclose(ok["b"], [1.0, 1.0])
    V = {k: np.random.default_rng(i).normal(size=20) for i, k in enumerate("abc")}
    o3 = pcgrad(V, rng)                                            # proyeksi tak pernah menambah norma
    assert all(np.linalg.norm(o3[k]) <= np.linalg.norm(V[k]) + 1e-12 for k in V)
    assert set(pcgrad({"a": g1}, rng)) == {"a"}


def test_ce_np_matches_autograd():
    z = np.random.default_rng(1).normal(size=256)
    assert abs(ce_np(z, 65) - float(cross_entropy(Tensor(z), 65).data)) < 1e-10


def test_span_target_semantics():
    """Sebelum pembaruan lambat pertama, zn = l2n(target_dict[byte]); target span = l2n(rata2 zn)."""
    ln, m = mk(slow_every=10 ** 6, replay=False)
    data = TEXT[:12]
    tgt = m.target_dict().copy()
    s, _ = feed(ln, m, data)
    for j in range(len(data)):
        assert np.allclose(s.buf[j].zn, l2n_np(tgt[data[j]]))
    i, h = 2, 4
    exp = l2n_np(np.mean([l2n_np(tgt[data[j]]) for j in range(i + 1, i + h + 1)], axis=0))
    assert np.allclose(ln._span_target(s, i, h), exp)


def test_replay_priority():
    ln, m = mk(replay_cap=3)
    s = Stream(m)

    def add(prio):
        s.seg, s.seg_n, s.seg_ce = [1, 2], 4, prio * 4
        ln._replay_add(s)
    for p in (1.0, 2.0, 3.0):
        add(p)
    add(0.5)                                        # lebih rendah dari semua: ditolak
    assert sorted(r["prio"] for r in ln.replay) == [1.0, 2.0, 3.0]
    add(5.0)                                        # menggantikan prioritas terendah
    assert sorted(r["prio"] for r in ln.replay) == [2.0, 3.0, 5.0]


def test_traces_zero_copy():
    ln, m = mk()
    s, _ = feed(ln, m, TEXT[:20])
    before = sum(np.abs(v).sum() for L in s.st.layers for v in L.S.values())
    assert before > 0
    z = s.st.zero_traces()
    assert all(np.abs(v).sum() == 0 for L in z.layers for v in L.S.values())
    assert np.array_equal(z.layers[0].h, s.st.layers[0].h)          # state h dipertahankan
    assert sum(np.abs(v).sum() for L in s.st.layers for v in L.S.values()) == before


# ------------------------------------------------------------------ invarian
def test_determinism():
    r = []
    for _ in range(2):
        ln, m = mk(seed=3)
        _, ces = feed(ln, m, TEXT[:150])
        r.append(ces)
    assert np.array_equal(r[0], r[1])


def test_slow_update_bookkeeping():
    ln, m = mk(slow_every=8, replay=False)
    e0, tp0 = m.E.data.copy(), m.tp["E"].copy()
    feed(ln, m, TEXT[:40])
    assert ln.stats["slow_updates"] == 5 and ln.t_adam == 5
    assert not np.array_equal(m.E.data, e0)
    assert not np.array_equal(m.tp["E"], tp0)                      # EMA target ikut bergerak
    assert np.abs(m.tp["E"] - m.E.data).max() < np.abs(e0 - m.E.data).max() + 1e-12  # tertinggal di belakang
    assert all(np.isfinite(p.data).all() for p in m.params.values())
    assert set(ln.unc) == set(W_TASKS) and all(abs(v) <= 3.0 for v in ln.unc.values())


def test_eval_does_not_learn():
    ln, m = mk()
    feed(ln, m, TEXT[:120])
    before, adam0, t0 = snap(m), {k: [x.copy() for x in v] for k, v in ln.adam.items()}, ln.t_adam
    c1, c2 = ln.eval_stream(TEXT[:60]), ln.eval_stream(TEXT[:60])
    assert np.array_equal(c1, c2) and len(c1) == 59
    assert same(before, snap(m)) and ln.t_adam == t0
    assert all(np.array_equal(ln.adam[k][0], adam0[k][0]) for k in adam0)
    g = ln.generate(TEXT[:20], 25, temp=0.7, seed=1)
    assert isinstance(g, bytes) and len(g) == 25 and g == ln.generate(TEXT[:20], 25, temp=0.7, seed=1)
    assert same(before, snap(m))


def test_sleep_isolation():
    """Tidur: jalur cepat mati (adaptor tak berubah), jalur lambat jalan, aliran hidup tak tersentuh."""
    ln, m = mk(sleep_every=10 ** 9)                                # cegah tidur otomatis
    s, _ = feed(ln, m, TEXT[:100])
    assert len(ln.replay) > 0
    fast0 = {p.name: p.data.copy() for p in ln.fast}
    slow0 = m.E.data.copy()
    h0, t0, nb, su = s.st.layers[-1].h.copy(), s.t, ln.n_bytes, ln.stats["slow_updates"]
    ln.sleep()
    assert all(np.array_equal(p.data, fast0[p.name]) for p in ln.fast)
    assert not np.array_equal(m.E.data, slow0) and ln.stats["slow_updates"] > su
    assert np.array_equal(s.st.layers[-1].h, h0) and s.t == t0 and ln.n_bytes == nb
    assert ln.stats["sleeps"] == 1


def test_fast_path_updates_adapter_only_while_awake():
    ln, m = mk(slow_every=10 ** 6, replay=False)                   # tanpa pembaruan lambat
    slow0 = {k: p.data.copy() for k, p in m.params.items() if not p.fast}
    fast0 = {p.name: p.data.copy() for p in ln.fast}
    feed(ln, m, TEXT[:60])
    assert any(not np.array_equal(p.data, fast0[p.name]) for p in ln.fast)      # adaptor bergerak
    assert all(np.array_equal(m.params[k].data, v) for k, v in slow0.items())   # bobot lambat diam


def test_think_off_and_on():
    ln, m = mk(tk=dict(think_prob=0.0), replay=False)
    s, _ = feed(ln, m, TEXT[:80])
    assert ln.stats["depth"][0] == 80 and ln.stats["depth"][1:].sum() == 0
    assert ln.ema_loss["stop"] is None and ln.unc["stop"] == 0.0
    ln, m = mk(tk=dict(think_prob=1.0), replay=False)
    s, _ = feed(ln, m, TEXT[:80])
    assert ln.stats["depth"].sum() == 80
    assert ln.ema_loss["stop"] is not None and np.isfinite(ln.ema_loss["stop"])
    e = s.buf[s.t]
    assert len(e.chain) == m.cfg.think_max + 1 and abs(e.halt.sum() - 1) < 1e-9
    assert e.chain_state is s.st                                   # state yang dibawa = kedalaman terpilih


def test_state_cap_and_bounds():
    ln, m = mk(state_cap=0.05, state_w=1.0, replay=False)
    rnd = np.random.default_rng(0).integers(0, 256, 300, dtype=np.uint8).tobytes()
    s, ces = feed(ln, m, rnd)
    assert ln.ema_loss["state"] is not None                        # penalti batas aktif
    assert all(np.isfinite(L.h).all() and np.abs(L.h).max() < 50 for L in s.st.layers)
    assert np.isfinite(ces).all()


def test_checkpoint_roundtrip():
    ln, m = mk()
    feed(ln, m, TEXT[:200])
    path = os.path.join(tempfile.mkdtemp(), "ck.npz")
    ln.save(path)
    ln2 = Learner.load(path, TrainOpts(seed=0))
    assert same(snap(m), snap(ln2.m))
    assert all(np.array_equal(m.tp[k], ln2.m.tp[k]) for k in m.tp)            # EMA target tersimpan
    assert all(np.array_equal(a.Wk, b.Wk) and np.array_equal(a.Wv, b.Wv) for a, b in zip(m.layers, ln2.m.layers))
    assert ln2.unc == ln.unc and ln2.t_adam == ln.t_adam
    assert all(np.array_equal(ln.adam[k][i], ln2.adam[k][i]) for k in ln.adam for i in (0, 1))
    assert np.array_equal(ln.eval_stream(TEXT[:50]), ln2.eval_stream(TEXT[:50]))
    assert ln.generate(TEXT[:16], 20, seed=2) == ln2.generate(TEXT[:16], 20, seed=2)
    assert ln2.m.cfg == m.cfg


# ------------------------------------------------------------------ belajar
def test_learns_context_beyond_unigram():
    """Pada teks periodik, CE prekuensial seperempat akhir harus di bawah entropi unigram:
    artinya model memakai konteks (state/memori), bukan hanya frekuensi byte."""
    data = TEXT[:4500]
    p = np.bincount(np.frombuffer(data, np.uint8), minlength=256) / len(data)
    h0 = -(p[p > 0] * np.log2(p[p > 0])).sum()
    ln, m = mk(seed=0, d=32, d_mem=8)
    _, ces = feed(ln, m, data)
    bits = ces / np.log(2)
    last = bits[-len(bits) // 4:].mean()
    print(f"  seperempat akhir = {last:.3f} bit  |  entropi unigram = {h0:.3f} bit")
    assert bits[:200].mean() > last + 1.0          # menurun nyata
    assert last < h0


if __name__ == "__main__":
    import time
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            t0 = time.time(); print(name, end=" ", flush=True); fn(); print(f"({time.time()-t0:.1f}s)")
    print("learner ok")
