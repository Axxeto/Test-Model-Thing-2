"""Validasi gradien BPTT kontrol terhadap beda-hingga (kebenaran sebelum dipakai).

Tujuan: pastikan `layer_rec` + `head_logits` di `bptt_control.py` menghasilkan gradien yang
BENAR. Bila kontrol ini salah, perbandingan "arsitektur vs algoritma" tidak sah.

Jebakan yang sudah diketahui (lihat LAPORAN §12.2):
  * arena/alias: jangan mengganti `.data` dengan objek baru (`p.data = w` memutus view);
    di sini parameter TMT2 berdiri sendiri, jadi cukup `.data[:] = w`.
  * jangan membaca `gacc` setelah satu run penuh (di sini tidak dipakai).
  * FD harus memakai state yang sama: hitung ulang maju penuh untuk tiap perturbasi.

Pakai: python scripts/bptt_gradcheck.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tmt2.autograd import Tensor, take, detach, cross_entropy
from tmt2.model import Config, TMT2
from scripts.bptt_control import layer_rec, layer_ff, head_logits

LN2 = np.log(2)


def loss_of(model, ids, params_names):
    """Loss CE total pada satu chunk (state mulai nol)."""
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
        logits = head_logits(model, y)
        tot = tot + cross_entropy(logits, nb)
        h = hn          # PENTING: JANGAN detach di dalam chunk -> biarkan graf mengalir
    return tot          # (detach hanya di BATAS chunk, itulah "terpotong" pada BPTT)


def main():
    rng = np.random.default_rng(0)
    V, T = 6, 8
    ids = rng.integers(0, V, T)
    cfg = Config(d=8, vocab=V, n_layers=1, d_mem=8, use_memory=False,
                 head="linear", adapter_rank=0, use_innovation=False, seed=3)
    model = TMT2(cfg)

    # nama parameter yang diuji (campuran rekurensi + kepala)
    targets = ["L0.Wa", "L0.B", "L0.ba", "L0.Wc", "L0.Wo", "E", "W_out", "byte_bias"]
    eps = 1e-6

    # ---- gradien analitik
    # CATATAN: Param mengakumulasi ke `p.tg[CTX.task]`, BUKAN `p.grad` (lihat autograd.Param).
    from tmt2.autograd import CTX
    for p in model.params.values():
        p.zero_grad()
    loss = loss_of(model, ids, targets)
    loss.backward()
    ana = {n: model.params[n].tg[CTX.task].copy() for n in targets}

    # ---- beda-hingga terpusat
    print("param        maks |err| rel   (oracle)")
    worst = 0.0
    for n in targets:
        p = model.params[n]
        base = p.data.copy()
        g = np.zeros_like(base)
        it = np.nditer(base, flags=["multi_index"])
        # batasi jumlah elemen untuk kecepatan
        cnt = 0
        while not it.finished and cnt < 24:
            idx = it.multi_index
            x0 = base[idx]
            p.data[idx] = x0 + eps
            lp = float(loss_of(model, ids, targets).data)
            p.data[idx] = x0 - eps
            lm = float(loss_of(model, ids, targets).data)
            p.data[idx] = x0
            g[idx] = (lp - lm) / (2 * eps)
            cnt += 1
            it.iternext()
        a = ana[n].reshape(-1)[:cnt]
        gn = g.reshape(-1)[:cnt]
        # galat RELATIF terhadap skala gradien (bukan per-elemen, beberapa elemen ~0)
        rel = np.max(np.abs(a - gn)) / (np.max(np.abs(gn)) + 1e-8)
        worst = max(worst, rel)
        print(f"{n:12s} {rel:.3e}")
    print(f"\nMAKS rel err (skala gradien) = {worst:.3e}")
    print("HASIL:", "LULUS" if worst < 1e-4 else "GAGAL")


if __name__ == "__main__":
    main()
