"""AUDIT KEBOCORAN tabel mixer (`_mix_ng`) — uji hipotesis telaah eksternal.

Hipotesis (telaah): angka mixer 1,138 bit (warm/cold) TIDAK SAH karena jendela uji
bocor ke tabel n-gram `_mix_ng` lewat eval ONLINE yang dijalankan SEBELUM eval beku
pada jendela yang SAMA.

Mekanisme di kode (`fast_learner.step`):
    if train and s.online and prev is not None:
        ngm.update(tuple(s.hist)[-ngm.order:], int(b))     # <- hanya saat online
`run_variant` default modes = ["online","warm","cold"] -> online dulu (tabel menyerap
jendela uji sampai orde 8), lalu warm/cold membaca tabel yang sudah berisi jendela itu.

Empat uji, semua < ~2 menit pada data kecil:
  T1 GUARD   : jumlah entri tabel SEBELUM vs SESUDAH eval_stream(beku) harus identik.
               (Bukti langsung: eval beku tidak menyentuh tabel.)
  T2 ORDER   : angka beku bila dijalankan SEBELUM online vs SESUDAH online.
               (Bila hipotesis benar: sesudah >> sebelum, khususnya orde tinggi.)
  T3 WINDOW  : angka beku pada jendela UJI vs jendela BARU yang belum disentuh learner.
  T4 PPM     : mixer dilatih pada train; dieval beku pada test. Bandingkan dengan
               PPM orde-8 MURNI (OnlineNgramN orde 8, dilatih pada train yang sama,
               dieval prekuensial pada test) -> baseline adil tanpa SSM.
"""
import argparse, json, os, sys, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tmt2.model import Config, TMT2
from tmt2.learner import TrainOpts, Stream
from tmt2.fast_learner import FastLearner, OnlineNgramN
from tmt2.corpus import load, sanity_check

LN2 = float(np.log(2))


def table_size(ng):
    """(total entri konteks, total pasangan (konteks->simbol), total count)."""
    nctx = sum(len(t) for t in ng.tabs)
    npair = sum(len(d) for t in ng.tabs for d, _ in t.values())
    ncnt = sum(n for t in ng.tabs for _, n in t.values())
    return nctx, npair, float(ncnt)


def ce_bits(ce_sum, n):
    return ce_sum / max(n, 1) / LN2


def eval_frozen(fl, m, data, carry_stream=None):
    """Eval BEKU (train=False). Bila carry_stream diberi, state dibawa dari sana (warm);
    jika None, stream baru (cold). Tidak menyentuh tabel."""
    s = carry_stream if carry_stream is not None else Stream(m, online=False)
    ce, n = 0.0, 0
    for b in data:
        c = fl.step(s, int(b), train=False, think=False)
        if c is not None:
            ce += float(c); n += 1
    return ce, n


def train_online(fl, m, data):
    s = Stream(m, online=True)
    for b in data:
        fl.step(s, int(b), train=True, think=False)
    return s


def build(V, seed, orders, d=32, d_mem=8):
    cfg = Config(vocab=V, d=d, n_layers=2, d_mem=d_mem, head="linear", deep_grad=True,
                 lr=1e-2, warmup=200, slow_every=8, seg_len=256, replay=False,
                 sleep_every=10 ** 9, seed=seed, mixer_k=1, mixer_match=True,
                 mixer_lr_mult=0.1, mixer_match_orders=orders)
    opts = TrainOpts(seed=seed, think_prob=0.0, only_byte_loss=True, use_stop=False)
    m = TMT2(cfg)
    return FastLearner(m, opts), m


def ppmg_ce(train_bytes, test_bytes, order=8, V=256):
    """PPM/n-gram MURNI: OnlineNgramN orde `order`, dilatih pada train lalu dieval
    prekuensial BEKU (tanpa update) pada test. Konteks = riwayat penuh."""
    ng = OnlineNgramN(V, order)
    hist = []
    for b in train_bytes:
        ng.update(tuple(hist)[-order:], int(b))
        hist.append(int(b))
    # eval prekuensial: pakai logprior_order(ctx, order) sebagai prediksi tunggal
    hist_t = list(hist[-order:])
    ce, n = 0.0, 0
    for b in test_bytes:
        lp = ng.logprior_order(tuple(hist_t), order)
        ce += -lp[int(b)] / LN2
        n += 1
        hist_t.append(int(b))
        hist_t = hist_t[-order:]
    return ce / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="wikitext")
    ap.add_argument("--train", type=int, default=40000)
    ap.add_argument("--test", type=int, default=4000)
    ap.add_argument("--orders", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/audit_leak.jsonl")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    raw = load(a.corpus)[:a.train + a.test + 8000]
    sanity_check(raw, vocab=256, name=a.corpus, expect_len=len(raw))
    ids = np.frombuffer(raw, dtype=np.uint8).astype(np.int64)
    tr = ids[:a.train]
    te = ids[a.train:a.train + a.test]
    fresh = ids[a.train + a.test:a.train + a.test + a.test]   # jendela BARU, belum disentuh
    V = 256
    print(f"=== audit_leak {a.corpus} seed={a.seed} orders={a.orders} "
          f"ntrain={len(tr)} ntest={len(te)} ===")

    fl, m = build(V, a.seed, a.orders)
    s_tr = train_online(fl, m, tr)
    print(f"[latih] {len(tr)} byte; tabel: {table_size(fl._mix_ng)}")

    # ---------- T1 GUARD: eval beku tidak boleh mengubah tabel ----------
    before = table_size(fl._mix_ng)
    _ = eval_frozen(fl, m, te)                       # cold, beku
    after_cold = table_size(fl._mix_ng)
    _ = eval_frozen(fl, m, te, carry_stream=s_tr)    # warm, beku
    after_warm = table_size(fl._mix_ng)
    t1_ok = (before == after_cold == after_warm)
    print(f"[T1 GUARD] sebelum={before}  sesudah_cold={after_cold}  "
          f"sesudah_warm={after_warm}  -> {'LULUS (tabel tak tersentuh)' if t1_ok else 'GAGAL (tabel berubah!)'}")

    # ---------- T2 ORDER: beku SEBELUM online vs SESUDAH online ----------
    fl2, m2 = build(V, a.seed, a.orders)
    s2 = train_online(fl2, m2, tr)
    c_pre, n_pre = eval_frozen(fl2, m2, te, carry_stream=s2)   # BEKU DULU
    pre_bits = ce_bits(c_pre, n_pre)
    # sekarang online pada jendela yang sama (menyerapnya), lalu beku lagi
    for b in te:
        fl2.step(s2, int(b), train=True, think=False)
    c_post, n_post = eval_frozen(fl2, m2, te, carry_stream=s2)  # BEKU SESUDAH
    post_bits = ce_bits(c_post, n_post)
    print(f"[T2 ORDER] beku-SEBELUM-online={pre_bits:.4f}  beku-SESUDAH-online={post_bits:.4f}  "
          f"selisih={post_bits - pre_bits:+.4f} bit")

    # ---------- T3 WINDOW: beku pada jendela BARU (belum disentuh) ----------
    c_fresh, n_fresh = eval_frozen(fl2, m2, fresh, carry_stream=s2)
    fresh_bits = ce_bits(c_fresh, n_fresh)
    print(f"[T3 WINDOW] beku jendela-baru={fresh_bits:.4f} (ukuran {len(fresh)})  "
          f"vs jendela-uji-tercemar={post_bits:.4f}")

    # ---------- T4 PPM murni orde-8 pada split yang sama ----------
    ppm_bits = ppmg_ce(tr, te, order=8, V=V)
    print(f"[T4 PPM] PPM/ngram orde-8 murni (latih=train, eval prekuensial beku di test) = {ppm_bits:.4f} bit")

    row = {"corpus": a.corpus, "seed": a.seed, "orders": a.orders,
           "ntrain": len(tr), "ntest": len(te),
           "t1_table_before": before, "t1_table_after_cold": after_cold,
           "t1_table_after_warm": after_warm, "t1_ok": bool(t1_ok),
           "t2_frozen_before_online": round(pre_bits, 4),
           "t2_frozen_after_online": round(post_bits, 4),
           "t2_delta": round(post_bits - pre_bits, 4),
           "t3_frozen_fresh_window": round(fresh_bits, 4),
           "t4_ppm_order8": round(ppm_bits, 4)}
    with open(a.out, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    print("\n->", a.out)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
