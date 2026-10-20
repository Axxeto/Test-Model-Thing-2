"""Kontrol TRUNCATED BPTT vs RTRL diagonal — memisahkan ARSITEKTUR dari ALGORITMA BELAJAR.

Pertanyaan (Langkah 3 telaah eksternal):
  SSM tidak menambah apa pun di atas mixer pada rezim yang diuji. Apakah itu karena
  ARSITEKTURNYA (SSM memang tidak menangkap struktur berguna), atau karena ALGORITMA
  BELAJARNYA (jejak RTRL diagonal terpotong: gradien antar-lapisan secara temporal hilang,
  lihat LAPORAN §9)?

Cara: pakai arsitektur yang SAMA PERSIS (kelas `TMT2` yang sama, persamaan `layer_step`
yang sama), tetapi hitung gradien rekurensi dengan BPTT terpotong (unroll satu chunk,
chunk 32-64, 1 lapisan bila diminta) dalam autograd murni. Lalu latih dan evaluasi dengan
protokol yang sama seperti `ablation_mixer.py`.

Catatan kejujuran:
  * Ini bukan reimplementasi jalur cepat: ia memang lebih lambat (autograd + unroll).
    Tujuannya BUKAN kecepatan, melainkan gradien rekurensi yang benar.
  * Bila BPTT > RTRL secara material -> hambatannya ALGORITMA (jejak terpotong), dan P1
    (kernel terpadu / BPTT chunk) jadi berharga.
  * Bila BPTT ~= RTRL -> hambatannya ARSITEKTUR/pemodelan, bukan pelacakan gradien.

Pakai:
  python scripts/bptt_control.py --kind markov_det --chunk 32 --seeds 1,2,3
  python scripts/bptt_control.py --kind wikitext --ntrain 100000 --ntest 20000
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tmt2.autograd import (Tensor, matmul, layer_norm, l2norm, silu, transpose, take,
                           cross_entropy, concat0, detach)
from tmt2.model import Config, TMT2, ln_np, l2n_np
from tmt2.corpus import load, markov_det_stream, markov_randcol_stream, sanity_check

LN2 = float(np.log(2))


# ----------------------------------------------------------------- data
def get_data(kind, ntrain, ntest, seed):
    n = ntrain + ntest
    if kind == "markov_det":
        ids, floor = markov_det_stream(n, 2, 8, 0.10, seed)
        return ids, 8, float(floor), f"markov det k=2 V=8 (lantai {floor:.3f})"
    if kind == "markov_randcol":
        ids, floor = markov_randcol_stream(n, 2, 8, 0.10, seed)
        return ids, 8, float(floor), f"markov randcol k=2 V=8 (lantai {floor:.3f})"
    raw = load(kind)[:n]
    sanity_check(raw, vocab=256, name=kind, expect_len=len(raw))
    return np.frombuffer(raw, dtype=np.uint8).astype(np.int64), 256, None, kind


# ----------------------------------------------------------------- forward (1 lapisan, autograd)
def layer_ff(cfg, L, x, h, M):
    """h_t -> y_t, sama dengan core.ff_readout (tanpa memori bila cfg.use_memory False)."""
    hn = layer_norm(h)
    c = matmul(hn, L.Wc)
    g = silu(matmul(c, L.Wg1)) * matmul(c, L.Wg2)
    f = matmul(g, L.Wo)
    return x + f


def layer_rec(cfg, L, x, h_prev):
    """h_t = a*h_{t-1} + (1-a)*(x B); a = sigmoid(x Wa + ba). Mengembalikan Tensor h_t.

    x dan h_prev adalah Tensor (daun bila perlu) -> graf mengalir lewat keduanya, jadi
    BPTT menangkap kopling TEMPORAL dan ANTAR-LAPISAN yang jejak RTRL diagonal lewatkan.
    """
    a = _sigmoid(matmul(x, L.Wa) + L.ba)
    u = matmul(x, L.B)
    one = Tensor(np.array(1.0))
    return a * h_prev + (one - a) * u


def _sigmoid(t):
    from tmt2.autograd import sigmoid
    return sigmoid(t)


def head_logits(model, p):
    """Kepala linear: logits = W_out @ l2norm(p) + byte_bias (sama dengan byte_logits)."""
    return matmul(l2norm(p), transpose(model.W_out)) + model.byte_bias


# ----------------------------------------------------------------- train chunk (BPTT)
def train_chunk_bptt(model, ids, params, chunk, n_layers_used):
    """Latih satu chunk dengan BPTT terpotong. Gradien diakumulasi di `p.tg[CTX.task]`.

    `state` (h per lapisan) dibawa ANTAR chunk tetapi di-`detach` di batas chunk (itulah
    "terpotong"). DI DALAM chunk graf tetap hidup sehingga BPTT menangkap kopling temporal
    dan antar-lapisan yang jejak RTRL diagonal lewatkan.
    """
    m, cfg = model, model.cfg
    d = cfg.d
    h = [Tensor(np.zeros(d)) for _ in range(cfg.n_layers)]     # state awal chunk (detached)
    ce, n = 0.0, 0
    # bersihkan akumulasi lama supaya gradien chunk ini murni
    for p in params:
        p.tg = {}
    for i, b in enumerate(ids):
        x = take(m.E, int(b))
        # majukan tumpukan (di sini 1 lapisan diuji secara default; n_layers_used menentukan)
        hs = []
        cur = x
        for l in range(n_layers_used):
            hn = layer_rec(cfg, m.layers[l], cur, h[l])
            hs.append(hn)
            cur = layer_ff(cfg, m.layers[l], cur, hn, None)
        # puncak -> logits -> CE terhadap byte BERIKUTNYA
        if i + 1 < len(ids):
            nb = int(ids[i + 1])
            logits = head_logits(m, cur)
            ce_t = cross_entropy(logits, nb)
            ce_t.backward()
            ce += float(ce_t.data)
            n += 1
        # PENTING: diputus dari graf HANYA di batas chunk (di luar fungsi ini). Di dalam chunk
        # graf tetap hidup supaya BPTT menangkap kopling temporal. Detach tiap langkah akan
        # menurunkan BPTT menjadi backprop satu-langkah dan gradien rekurensi jadi SALAH.
        h = hs
    return ce, n


# ----------------------------------------------------------------- alternatif: SGD atas paramet
def sgd_step(params, lr):
    """SGD satu langkah. Param menyimpan gradien di `tg[CTX.task]`, bukan `.grad`."""
    from tmt2.autograd import CTX
    for p in params:
        g = p.tg.get(CTX.task)
        if g is None:
            continue
        p.data -= lr * np.clip(g, -1.0, 1.0)
        p.tg[CTX.task] = None
    for p in params:
        p.tg.pop(CTX.task, None)


def eval_bptt(model, cfg, test_ids, n_layers):
    """CE prekuensial bit/simbol pada data uji, tanpa update, state mulai nol."""
    h = [Tensor(np.zeros(cfg.d)) for _ in range(n_layers)]
    ce, n = 0.0, 0
    for i, b in enumerate(test_ids):
        if i + 1 >= len(test_ids):
            break
        nb = int(test_ids[i + 1])
        x = take(model.E, int(b))
        hs, cur = [], x
        for l in range(n_layers):
            hn = layer_rec(cfg, model.layers[l], cur, h[l])
            hs.append(hn); cur = layer_ff(cfg, model.layers[l], cur, hn, None)
        logits = head_logits(model, cur)
        p = _softmax(logits.data)
        ce += -np.log2(max(p[nb], 1e-12))
        n += 1
        h = [detach(t) for t in hs]
    return ce / max(n, 1)


def run_bptt(cfg, train_ids, test_ids, args, sd):
    """Latih dengan BPTT terpotong; kembalikan (bpb, detik)."""
    model = TMT2(cfg)
    params = []
    for l in range(args.n_layers):
        L = model.layers[l]
        params += [L.Wa, L.ba, L.B, L.Wc, L.Wg1, L.Wg2, L.Wo]
    params += [model.E, model.W_out, model.byte_bias]
    t0 = time.time()
    for ep in range(args.epochs):
        pos = 0
        while pos < len(train_ids):
            seg = train_ids[pos:pos + args.chunk]
            if len(seg) < 2:
                break
            train_chunk_bptt(model, seg, params, args.chunk, args.n_layers)
            sgd_step(params, args.lr)
            pos += args.chunk
    bpb = eval_bptt(model, cfg, test_ids, args.n_layers)
    return bpb, time.time() - t0


def run_rtrl(cfg, train_ids, test_ids, args, sd):
    """Arm pembanding: RTRL diagonal, mesin proyek yang sesungguhnya (FastLearner).

    Konfigurasi dibuat sedekat mungkin dengan arm BPTT (d, n_layers, no memori, no inovasi,
    head linear) dan dijalankan dalam mode ONLINE prekuensial pada jendela uji yang sama.
    """
    from tmt2.learner import TrainOpts, Stream
    from tmt2.fast_learner import FastLearner
    from tmt2.model import Config as Cfg
    cfg2 = Cfg(d=args.d, vocab=cfg.vocab, n_layers=args.n_layers, d_mem=8,
               use_memory=False, head="linear", adapter_rank=0, use_innovation=False,
               deep_grad=(args.n_layers > 1), lr=args.lr, warmup=1)
    m = TMT2(cfg2)
    fl = FastLearner(m, TrainOpts(only_byte_loss=True, think_prob=0.0))
    t0 = time.time()
    # latih (online) pada data latih. JUMLAH PASS disamakan dengan arm BPTT (`--epochs`);
    # kalau tidak, perbandingan tidak adil (BPTT mendapat 5 pass, RTRL 1 pass).
    for ep in range(args.epochs):
        s = Stream(m, online=True)
        for b in train_ids:
            fl.step(s, int(b), train=True, think=False)
    # evaluasi prekuensial pada jendela uji bila bobot beku (cek murni generalisasi)
    s2 = Stream(m, online=False)
    ce, n = 0.0, 0
    for b in test_ids:
        c = fl.step(s2, int(b), train=False, think=False)
        if c is not None:
            ce += float(c) / LN2; n += 1
    return ce / max(n, 1), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="markov_det")
    ap.add_argument("--ntrain", type=int, default=4000)
    ap.add_argument("--ntest", type=int, default=2000)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--n_layers", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seeds", default="1")
    ap.add_argument("--arms", default="bptt,rtrl")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    results = []
    for sd in seeds:
        ids, V, floor, desc = get_data(args.kind, args.ntrain, args.ntest, sd)
        train_ids = ids[:args.ntrain]
        test_ids = ids[args.ntrain:]
        cfg = Config(d=args.d, vocab=V, n_layers=args.n_layers, d_mem=8,
                     use_memory=False, head="linear", adapter_rank=0, use_innovation=False,
                     seed=sd)
        rec = dict(seed=sd, kind=args.kind, desc=desc, floor=floor, d=args.d,
                   n_layers=args.n_layers, chunk=args.chunk, lr=args.lr, epochs=args.epochs,
                   ntrain=args.ntrain, ntest=args.ntest)
        line = f"seed {sd} [{desc} lantai={floor if floor is None else round(floor,3)}] "
        if "bptt" in arms:
            bpb, ts = run_bptt(cfg, train_ids, test_ids, args, sd)
            rec["bptt_bpb"] = bpb; rec["bptt_s"] = ts
            line += f"BPTT={bpb:.4f}({ts:.0f}s) "
        if "rtrl" in arms:
            bpb, ts = run_rtrl(cfg, train_ids, test_ids, args, sd)
            rec["rtrl_bpb"] = bpb; rec["rtrl_s"] = ts
            line += f"RTRL={bpb:.4f}({ts:.0f}s) "
        results.append(rec)
        print(line)

    if args.out:
        with open(args.out, "a") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print("ditulis ke", args.out)
    for arm in ("bptt", "rtrl"):
        vals = [r[f"{arm}_bpb"] for r in results if f"{arm}_bpb" in r]
        if vals:
            print(f"RATA-RATA {arm.upper()} bpb = {np.mean(vals):.4f}")


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


if __name__ == "__main__":
    main()
