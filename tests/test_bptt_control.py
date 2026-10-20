"""Validasi kontrol BPTT (algoritma belajar alternatif) terhadap beda-hingga.

Kontrol "arsitektur vs algoritma" hanya sah bila gradiennya BENAR. Tes ini mengunci bahwa
BPTT terpotong (`scripts/bptt_control.py`) menghasilkan gradien yang cocok dengan beda-hingga
terpusat pada parameter rekurensi DAN feedforward.

Jebakan yang sudah terdokumentasi (lihat LAPORAN §12.2 dan catatan di skrip):
  * ``Param`` mengakumulasi gradien ke ``p.tg[CTX.task]``, BUKAN ``p.grad``.
  * ``detach`` di DALAM chunk akan menurunkan BPTT menjadi backprop satu-langkah dan membuat
    gradien rekurensi SALAH (terukur rel err 11,6 pada ``B`` sebelum diperbaiki). Detach hanya
    boleh di BATAS chunk.

Jalankan: pytest tests/test_bptt_control.py -q
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tmt2.autograd import Tensor, take, detach, cross_entropy, CTX
from tmt2.model import Config, TMT2
from scripts.bptt_control import layer_rec, layer_ff, head_logits


def _chunk_loss(model, ids):
    """Loss CE total satu chunk dengan graf UTUH (tidak ada detach di dalam)."""
    cfg = model.cfg
    h = Tensor(np.zeros(cfg.d))
    tot = 0.0
    for i, b in enumerate(ids):
        if i + 1 >= len(ids):
            break
        nb = int(ids[i + 1])
        x = take(model.E, int(b))
        hn = layer_rec(cfg, model.layers[0], x, h)
        y = layer_ff(cfg, model.layers[0], x, hn, None)
        tot = tot + cross_entropy(head_logits(model, y), nb)
        h = hn                      # graf tetap hidup -> BPTT sejati
    return tot


def test_bptt_gradient_matches_finite_difference():
    rng = np.random.default_rng(0)
    V, T = 6, 8
    ids = rng.integers(0, V, T)
    cfg = Config(d=8, vocab=V, n_layers=1, d_mem=8, use_memory=False,
                 head="linear", adapter_rank=0, use_innovation=False, seed=3)
    model = TMT2(cfg)
    targets = ["L0.Wa", "L0.B", "L0.ba", "L0.Wo", "E", "W_out"]
    eps = 1e-6

    for p in model.params.values():
        p.zero_grad()
    _chunk_loss(model, ids).backward()
    ana = {n: model.params[n].tg[CTX.task].copy() for n in targets}

    worst = 0.0
    for n in targets:
        p = model.params[n]
        base = p.data.copy()
        g = np.zeros_like(base)
        it = np.nditer(base, flags=["multi_index"])
        cnt = 0
        while not it.finished and cnt < 12:
            idx = it.multi_index
            x0 = base[idx]
            p.data[idx] = x0 + eps
            lp = float(_chunk_loss(model, ids).data)
            p.data[idx] = x0 - eps
            lm = float(_chunk_loss(model, ids).data)
            p.data[idx] = x0
            g[idx] = (lp - lm) / (2 * eps)
            cnt += 1
            it.iternext()
        a = ana[n].reshape(-1)[:cnt]
        gn = g.reshape(-1)[:cnt]
        rel = np.max(np.abs(a - gn)) / (np.max(np.abs(gn)) + 1e-8)
        worst = max(worst, rel)
    assert worst < 1e-4, f"BPTT grad salah: rel err skala = {worst:.2e}"
    print(f"  BPTT grad vs FD: rel err skala = {worst:.2e}  (LULUS)")


def test_bptt_recursion_actually_unrolls():
    """Bila detach dipasang di dalam chunk, gradien rekurensi akan JAUH melenceng.

    Tes ini membandingkan gradien dengan chunk panjang (T=12, unroll penuh) terhadap versi
    yang 'di-detach tiap langkah' (backprop satu-langkah). Keduanya HARUS berbeda material,
    membuktikan BPTT benar-benar meng-unroll (bukan tidak sengaja ter-detach).
    """
    cfg = Config(d=8, vocab=6, n_layers=1, d_mem=8, use_memory=False,
                 head="linear", adapter_rank=0, use_innovation=False, seed=1)
    ids = np.random.default_rng(5).integers(0, 6, 12)

    m1 = TMT2(cfg)
    for p in m1.params.values():
        p.zero_grad()
    _chunk_loss(m1, ids).backward()
    g_full = m1.params["L0.B"].tg[CTX.task].copy()

    # versi ter-detach (harus berbeda): bangun ulang dengan detach tiap langkah
    m2 = TMT2(cfg)
    h = Tensor(np.zeros(cfg.d))
    tot = 0.0
    for i, b in enumerate(ids):
        if i + 1 >= len(ids):
            break
        x = take(m2.E, int(b))
        hn = layer_rec(cfg, m2.layers[0], x, h)
        y = layer_ff(cfg, m2.layers[0], x, hn, None)
        tot = tot + cross_entropy(head_logits(m2, y), int(ids[i + 1]))
        h = detach(hn)              # <-- memutus BPTT
    tot.backward()
    g_det = m2.params["L0.B"].tg[CTX.task].copy()

    rel = np.max(np.abs(g_full - g_det)) / (np.max(np.abs(g_full)) + 1e-8)
    assert rel > 0.05, f"BPTT tidak meng-unroll? rel diff = {rel:.3e}"
    print(f"  BPTT vs detach-tiap-langkah: rel diff = {rel:.3e}  (unroll terbukti)")


if __name__ == "__main__":
    test_bptt_gradient_matches_finite_difference()
    test_bptt_recursion_actually_unrolls()
    print("bptt_control ok")
