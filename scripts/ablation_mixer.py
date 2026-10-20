"""Ablasi Pilar 2 (mixer konteks) — protokol ADIL, tiga metrik, kriteria pra-terdaftar.

KOREKSI dari versi pertama (lihat LAPORAN §12, telaah eksternal):
  1. Generator Markov lama (`tab[ctx, randint]`) lantainya ~2,25-2,46 bit, BUKAN 0,30. Kini
     lantai dihitung dari tabel sebenarnya (`markov_det_stream`/`markov_randcol_stream`).
  2. Tren "0,04 -> 0,09 bit" pada versi lama TIDAK SAH: seed berbeda, jendela uji berbeda,
     1 seed/sel. Kini: jendela uji SAMA, >=3 seed, tiap sel dilaporkan per-seed.
  3. Varian "B mixer-saja" lama masih melatih `E` dan menyisakan kepala `W_out` acak -> bukan
     lantai murni. Kini ada B' (murni: `W_out=0`, `byte_bias=0`, hanya `mix_w` dilatih).
  4. C-residual: mulai TEPAT di titik B' (`W_out=0`, `mixer_init` dari B', LR SSM kecil)
     sehingga yang diukur adalah kontribusi MARGINAL SSM dari init yang benar.

Tiga metrik (semua pada JENDELA UJI YANG SAMA):
  * online   : tabel & bobot TERUS belajar selama jendela uji (protokol proyek sesungguhnya).
  * warm     : bobot beku, state dibawa dari fase latih (kontinu).
  * cold     : bobot beku, state SSM di-reset ke nol (murni generalisasi konteks).

KRITERIA KEPUTUSAN (pra-terdaftar, jangan diubah setelah melihat hasil):
  SSM dianggap BERGUNA bila `C_residual < B' - 0,02` bit di KETIGA metrik, konsisten di >=3 seed.
  Bila tidak terpenuhi -> "SSM tidak menambah apa pun di atas jalur konteks-lokal pada rezim ini".

KOREKSI 2 (audit kebocoran 2026-10-03, lihat LAPORAN §16):
  Urutan eval lama (online dulu, baru warm/cold) MENGONTAMINASI angka beku: eval online
  menyerap jendela uji ke tabel n-gram `_mix_ng` (sampai orde 8), sehingga eval beku
  berikutnya menghafalnya dan jatuh ~1 bit secara palsu (terukur -1,32 bit di WikiText,
  -0,99 bit di Indonesia, identik di 3 seed). Diperbaiki: default `--eval_order frozen_first`
  (eval beku dijalankan SEBELUM online). Audit: `scripts/audit_leak.py`.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from tmt2.model import Config, TMT2
from tmt2.learner import TrainOpts, Stream
from tmt2.fast_learner import FastLearner
from tmt2.corpus import (load, random_bytes, markov_det_stream, markov_randcol_stream,
                         sanity_check)

LN2 = float(np.log(2))


def get_data(kind, ntrain, ntest, seed, deterministic=True):
    """Mengembalikan (ids, V, floor_bit, deskripsi). Jendela uji = data[ntrain:ntrain+ntest]."""
    n = ntrain + ntest
    if kind == "markov_det":
        ids, floor = markov_det_stream(n, 2, 8, 0.10, seed)
        return ids, 8, float(floor), f"markov det k=2 V=8 eps=0.10 (lantai {floor:.3f} bit)"
    if kind == "markov_randcol":
        ids, floor = markov_randcol_stream(n, 2, 8, 0.10, seed)
        return ids, 8, float(floor), f"markov randcol k=2 V=8 eps=0.10 (lantai {floor:.3f} bit)"
    raw = load(kind)[:n]
    sanity_check(raw, vocab=256, name=kind, expect_len=len(raw))
    return np.frombuffer(raw, dtype=np.uint8).astype(np.int64), 256, None, kind


def evaluate(fl, m, data_win, mode):
    """CE rata-rata bit/simbol pada satu jendela uji. mode: online|warm|cold."""
    if mode == "online":
        s = fl.live if fl.live is not None else Stream(m, online=True)
        s.online = True
    else:
        s = Stream(m, online=False)
    ce, n = 0.0, 0
    train = (mode == "online")
    for b in data_win:
        c = fl.step(s, int(b), train=train, think=False)
        if c is not None:
            ce += float(c); n += 1
    return ce / max(n, 1) / LN2


def run_variant(data, V, cfg_kw, opts_kw, ntrain, ntest, seed, modes, warm_from=None,
                eval_order="frozen_first"):
    """Jalankan satu varian.

    ``warm_from``: dict nama-parameter -> nilai. Dipakai C-residual untuk MULAI TEPAT di titik
    B' yang sudah dilatih (bukan sekadar zero-init). Bila diberi, bobot disalin SEBELUM latih.

    ``eval_order`` -- PENTING (audit kebocoran 2026-10-03, lihat §16):
      * "frozen_first" (DEFAULT, SAH): mode beku (warm/cold) dieval pada tabel & bobot yang
        BELUM pernah melihat jendela uji, BARU online. Online menyerap jendela uji ke `_mix_ng`,
        jadi urutan ini satu-satunya yang menghasilkan angka beku jujur.
      * "online_first" (KONTAMINASI): urutan lama (online, warm, cold). Online lebih dulu
        membuat tabel menghafal jendela uji, sehingga warm/cold jatuh ~1 bit secara palsu.
        Hanya dipertahankan untuk MEMBUKTIKAN mekanisme kebocoran; JANGAN pakai untuk klaim.
    """
    assert eval_order in ("frozen_first", "online_first"), eval_order
    cfg = Config(**cfg_kw)
    opts = TrainOpts(seed=seed, **opts_kw)
    m = TMT2(cfg)
    if warm_from:
        for k, v in warm_from.items():
            if k in m.params:
                m.params[k].data[:] = np.asarray(v).reshape(m.params[k].data.shape)
    fl = FastLearner(m, opts)
    tr, te = data[:ntrain], data[ntrain:ntrain + ntest]
    s = Stream(m, online=True)
    t0 = time.time()
    for b in tr:
        fl.step(s, int(b), train=True, think=False)
    train_s = time.time() - t0
    out = {"train_s": round(train_s, 1)}

    def do_frozen(mode):
        if mode == "cold":
            s2 = Stream(m, online=False)
            ce, n = 0.0, 0
            for b in te:
                c = fl.step(s2, int(b), train=False, think=False)
                if c is not None:
                    ce += float(c); n += 1
            out["cold"] = round(ce / max(n, 1) / LN2, 4)
        else:                                            # warm: state dibawa dari fase latih
            ce, n = 0.0, 0
            for b in te:
                c = fl.step(s, int(b), train=False, think=False)
                if c is not None:
                    ce += float(c); n += 1
            out["warm"] = round(ce / max(n, 1) / LN2, 4)

    def do_online():
        ce, n = 0.0, 0
        for b in te:
            c = fl.step(s, int(b), train=True, think=False)
            if c is not None:
                ce += float(c); n += 1
        out["online"] = round(ce / max(n, 1) / LN2, 4)

    frozen = [mm for mm in modes if mm in ("warm", "cold")]
    has_online = "online" in modes
    if eval_order == "frozen_first":
        for mm in frozen:                                # beku DULU (tabel masih bersih)
            do_frozen(mm)
        if has_online:
            do_online()
    else:                                                # online_first: urutan lama (tercemar)
        if has_online:
            do_online()
        for mm in frozen:
            do_frozen(mm)
    out["_params"] = {k: m.params[k].data.copy() for k in ("mix_w",) if k in m.params}
    return out


def variants_for(base_cfg, base_opts, d_ssm_lr, mixer_init):
    """Definisi varian. `base_cfg` TIDAK termasuk mixer/head; `base_opts` TIDAK termasuk only_params."""
    ssm_lr = base_cfg["lr"]
    v = {}
    # A: SSM-saja (tanpa mixer) -- referensi lama
    v["A_ssm_only"] = (dict(base_cfg, mixer_k=0), dict(base_opts))
    # B: mixer-saja "lama" (mix_w+E dilatih, kepala acak tetap ada) -- pembanding historis
    v["B_mixer_plus_E"] = (dict(base_cfg, mixer_k=1, mixer_match=True, mixer_init=mixer_init),
                           dict(base_opts, only_params=("mix_w", "E")))
    # B': mixer MURNI -- W_out=0, byte_bias=0, hanya mix_w dilatih (lantai sejati mixer).
    # `out_zero_init=True` -> logits awal HANYA dari mixer; byte_bias dipaksa nol oleh
    # `only_params=("mix_w",)` (tidak di-update) tetapi init-nya sudah nol.
    v["Bp_mixer_pure"] = (dict(base_cfg, mixer_k=1, mixer_match=True, mixer_init=mixer_init,
                               head="linear", out_zero_init=True),
                          dict(base_opts, only_params=("mix_w",)))
    # C: mixer+SSM penuh dari init normal
    v["C_mixer_ssm"] = (dict(base_cfg, mixer_k=1, mixer_match=True, mixer_init=mixer_init),
                        dict(base_opts))
    # C-residual: mulai TEPAT di B' (W_out=0), LR SSM kecil -> ukur kontribusi marginal murni.
    # SSM BOLEH bergerak (semua parameter dilatih), tetapi ia mulai dari titik di mana
    # logits = mixer saja. Jika ia mengalahkan B', itu kontribusi marginal jujur.
    v["Cres_mixer_ssm"] = (dict(base_cfg, mixer_k=1, mixer_match=True, mixer_init=mixer_init,
                                out_zero_init=True, lr=d_ssm_lr),
                           dict(base_opts))
    # D: prior n-gram lama (bukan mixer)
    v["D_ssm_ngram"] = (dict(base_cfg, mixer_k=0), dict(base_opts, ngram_lam=1.0))
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="wikitext",
                    help="wikitext|license|markov_det|markov_randcol")
    ap.add_argument("--train", type=int, default=200000)
    ap.add_argument("--test", type=int, default=20000)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--d_mem", type=int, default=8)
    ap.add_argument("--orders", default="1,2,3")
    ap.add_argument("--mixer_init", type=float, default=0.5)
    ap.add_argument("--d_ssm_lr", type=float, default=1e-3,
                    help="LR SSM untuk C-residual (kecil -> mulai dekat B')")
    ap.add_argument("--modes", default="online,warm,cold")
    ap.add_argument("--eval_order", default="frozen_first",
                    choices=["frozen_first", "online_first"],
                    help="frozen_first=SAH (beku sebelum online menyerap jendela uji); "
                         "online_first=urutan lama, TERKONTAMINASI (lihat §16)")
    ap.add_argument("--mixer_lr_mult", type=float, default=1.0,
                    help="pengali LR untuk mix_w (uji stabilitas online mixer murni)")
    ap.add_argument("--only", default="", help="daftar varian dipisah koma (kosong = semua)")
    ap.add_argument("--out", default="results/ablation_mixer2.jsonl")
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    modes = [m for m in a.modes.split(",") if m]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    for seed in seeds:
        data, V, floor, desc = get_data(a.corpus, a.train, a.test, seed)
        base_cfg = dict(vocab=V, d=a.d, n_layers=2, d_mem=a.d_mem, head="linear",
                        deep_grad=True, lr=1e-2, warmup=200, slow_every=8,
                        seg_len=256, replay=False, sleep_every=10 ** 9, seed=seed,
                        mixer_lr_mult=a.mixer_lr_mult,
                        mixer_match_orders=a.orders)
        base_opts = dict(think_prob=0.0, only_byte_loss=True, use_stop=False)
        vars_ = variants_for(base_cfg, base_opts, a.d_ssm_lr, a.mixer_init)
        if a.only:
            keep = set(a.only.split(","))
            vars_ = {k: v for k, v in vars_.items() if k in keep}

        print(f"\n=== {a.corpus} seed={seed}  {desc}  ntrain={a.train} ntest={a.test} "
              f"modes={modes} ===")
        row = {"corpus": a.corpus, "seed": seed, "d": a.d, "d_mem": a.d_mem,
               "ntrain": a.train, "ntest": a.test, "floor": floor, "desc": desc,
               "mixer_init": a.mixer_init, "orders": a.orders, "d_ssm_lr": a.d_ssm_lr,
               "eval_order": a.eval_order}
        # B' dijalankan DULU: bobot `mix_w` terlatihnya dipakai untuk warm-start C-residual,
        # sehingga C-residual BENAR-BENAR mulai di titik B' (bukan sekadar zero-init).
        bp_res = None
        bp_mix = None
        if "Bp_mixer_pure" in vars_:
            ck, ok = vars_["Bp_mixer_pure"]
            bp_res = run_variant(data, V, ck, ok, a.train, a.test, seed, modes,
                                 eval_order=a.eval_order)
            bp_mix = bp_res.pop("_params", {})
            row["Bp_mixer_pure"] = {k: v for k, v in bp_res.items()}
            ms = "  ".join(f"{mm}={bp_res.get(mm)}" for mm in modes)
            print(f"  {'Bp_mixer_pure':20s} {ms}  ({bp_res['train_s']}s)")

        for name, (ck, ok) in vars_.items():
            if name == "Bp_mixer_pure":
                continue
            try:
                warm = bp_mix if (name == "Cres_mixer_ssm" and bp_res is not None) else None
                res = run_variant(data, V, ck, ok, a.train, a.test, seed, modes, warm_from=warm,
                                  eval_order=a.eval_order)
                res.pop("_params", None)
                row[name] = res
                ms = "  ".join(f"{mm}={res.get(mm)}" for mm in modes)
                tag = " [warm-start dari B']" if warm else ""
                print(f"  {name:20s} {ms}  ({res['train_s']}s){tag}")
            except Exception as exc:                      # noqa: BLE001
                row[name] = {"err": f"{type(exc).__name__}: {exc}"}
                print(f"  {name:20s} GAGAL: {exc}")
        with open(a.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    # ---- ringkasan + evaluasi kriteria pra-terdaftar (per mode, rata-rata lintas seed)
    print("\n=== RINGKASAN (rata-rata lintas seed) ===")
    rows = [json.loads(l) for l in open(a.out, encoding="utf-8") if l.strip()]
    rows = [r for r in rows if r["corpus"] == a.corpus]
    for mode in modes:
        print(f"-- mode {mode}")
        agg = {}
        for r in rows:
            for k, v in r.items():
                if isinstance(v, dict) and mode in v:
                    agg.setdefault(k, []).append(v[mode])
        for k, vs in agg.items():
            print(f"   {k:20s} {np.mean(vs):.4f}  (n={len(vs)})")
        if "Bp_mixer_pure" in agg and "Cres_mixer_ssm" in agg:
            diffs = []
            for r in rows:
                bp, cr = r.get("Bp_mixer_pure", {}), r.get("Cres_mixer_ssm", {})
                if mode in bp and mode in cr:
                    diffs.append(bp[mode] - cr[mode])     # >0 berarti SSM menang
            if diffs:
                d = np.asarray(diffs)
                verdict = ("SSM BERGUNA" if (d > 0.02).all() and len(d) >= 3
                           else "SSM TIDAK menambah (<=0,02 bit)")
                print(f"   -> B' - C_residual per seed: {np.round(d,4).tolist()}  "
                      f"rata2={d.mean():+.4f}  KRITERIA: {verdict}")
    print("\n-> ", a.out)


if __name__ == "__main__":
    main()
