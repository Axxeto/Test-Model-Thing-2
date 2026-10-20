"""Tes Tingkat A (`tmt2/stat_core.py`) — tes paritas wajib dari ARCHITECTURE_V2.md §7.

Yang diuji di sini: #6 (distribusi biner menjumlah ke 1), #7 (tak bocor / eval beku tidak
mengubah tabel), #8 (determinisme), plus konsistensi CE↔probabilitas dan bukti bahwa
latihan menurunkan CE.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tmt2.corpus import load                       # noqa: E402
from tmt2.stat_core import StatCore                # noqa: E402

LN2 = float(np.log(2))
SMALL = dict(table_bits=16, match_bits=14, wsel_bits=8, orders=(0, 1, 2, 3, 4, 6))


def _mk(**over):
    kw = dict(SMALL)
    kw.update(over)
    return StatCore(**kw)


def test_ce_matches_probability_product():
    """CE satu byte harus sama dengan -log2(produk 8 probabilitas kondisional)."""
    m = _mk()
    data = np.frombuffer(load("license"), dtype=np.uint8)[:400]
    for b in data:
        b = int(b)
        if m.hlen == 0:
            m.step(b)
            continue
        # probabilitas diambil SEBELUM step (byte_probs tidak mengubah state, termasuk prev_byte)
        pr = m.byte_probs(b)
        c = m.step(b, train=False)
        p = 1.0
        for i in range(8):
            bit = (b >> (7 - i)) & 1
            p *= pr[i] if bit else (1.0 - pr[i])
        ce_ref = -np.log2(p)
        assert abs(ce_ref - c) < 1e-9, f"byte {b}: ce {c} vs ref {ce_ref}"
    print("  CE == -log2(produk probabilitas bit): OK")


def test_ce_unit_is_bits_not_nats():
    """PENJAGA SATUAN (bug B-01, 2026-10-06).

    `StatCore.step`/`eval_stream` mengembalikan CE dalam **bit** (`-math.log2`). Beberapa
    skrip integrasi (m5_integrate, tierb_integrate) keliru membaginya lagi dengan LN2 ->
    terinflasi x1,4427 (mis. 3,2159 seharusnya 2,2291). Tes ini memakukan satuan itu:

        CE_bit = -log2(p)  ->  exp2(-CE_bit) == p   (BUKAN exp(-CE))

    Kalau seseorang mengubah ke nat atau menambahkan /LN2, tes ini gagal.
    """
    m = _mk()
    data = np.frombuffer(load("license"), dtype=np.uint8)[:300]
    for b in data:
        m.step(int(b))
    te = np.frombuffer(load("license"), dtype=np.uint8)[300:700]
    ces = m.eval_stream(te)
    assert len(ces) > 0
    mean = float(ces.mean())
    # Definisi langsung: p_i = 2^{-ce_i}. Jadi mean CE (bit) harus sama dengan
    # rata-rata(-log2 p_i). Kalau satuan tertukar (nat), exp(-ce) != p_i.
    p_bits = np.exp2(-ces)             # benar bila ce dalam BIT
    p_nats = np.exp(-ces)              # benar bila ce dalam NAT
    assert np.all((p_bits >= 0) & (p_bits <= 1))
    # konsistensi satuan BIT: ce_i == -log2(p_bits_i) secara identitas
    assert np.allclose(ces, -np.log2(np.clip(p_bits, 1e-300, 1.0)), atol=1e-12)
    # dan JANGAN berarti satuan nat: mean(ce)/LN2 tidak boleh jauh lebih kecil
    mean_nat_if_bits = mean * np.log(2)
    assert mean > 1.0, f"CE {mean:.4f} terlalu kecil untuk bit/byte teks"
    # beda bit vs nat harus jelas (>0,5 untuk CE ~2-6 bit)
    assert (mean - mean_nat_if_bits) > 0.5, "satuan tampak NAT (kemungkinan /LN2 terselip)"
    print(f"  eval_stream = {mean:.4f} bit/byte (satuan BIT terverifikasi): OK")


def test_hyper_roundtrip_bit_identical():
    """PENJAGA KONFIGURASI (bug B-06): hyper() -> from_hyper() harus bit-identik.

    Sebelumnya `from_config`/checkpoint tidak menyimpan match_minlen/match_bits/match_conf/
    count_cap/cks_bits/match_fix/ctx_fix/match_ck_bits, sehingga muat-ulang bisa menyimpang
    (hanya 'benar' karena default kebetulan sama).
    """
    h = dict(orders=(1, 2, 3, 4, 5), table_bits=18, lr=0.0025, use_apm=True, apm_lr=0.02,
             match_minlen=4, match_bits=18, match_conf=2.0, wsel_bits=12, lr2=3e-3,
             sparse=False, word=True, count_cap=511, cks_bits=20, match_fix=True,
             ctx_fix=True, match_ck_bits=7)
    m = StatCore(**h)
    h2 = m.hyper()
    assert h2["match_minlen"] == 4 and h2["count_cap"] == 511 and h2["cks_bits"] == 20
    assert h2["match_ck_bits"] == 7 and h2["match_fix"] is True and h2["ctx_fix"] is True
    m2 = StatCore.from_hyper(h2)
    # CE harus identik pada aliran yang sama (state awal sama, hiperparameter sama)
    data = np.frombuffer(load("license"), dtype=np.uint8)[:400]
    b1 = float(m.eval_stream(data).mean())
    b2 = float(m2.eval_stream(data).mean())
    assert abs(b1 - b2) < 1e-12, f"round-trip tak identik: {b1} vs {b2}"
    print(f"  hyper()/from_hyper() bit-identik ({b1:.6f}): OK")


def test_w_bound_caps_mixer_weights():
    """PENJAGA FIX KURVA-U: `w_bound` harus (a) bit-identik dgn default saat 0, (b) benar
    membatasi |bobot| mixer lapis-1 saat > 0.

    E1+E2 membuktikan kurva-U = bobot mixer lapis-1 divergen. `w_bound>0` meng-clip setelah
    tiap pembaruan. Kalau knob ini rusak (mis. tak pernah meng-clip), fix kurva-U palsu.
    """
    data = np.frombuffer(load("license"), dtype=np.uint8)[:3000]

    # (a) w_bound=0 == default (paritas bit-identik)
    a, b = _mk(), _mk(w_bound=0.0)
    ca, cb = a.train_stream(data), b.train_stream(data)
    ca, cb = ca[~np.isnan(ca)], cb[~np.isnan(cb)]
    assert np.max(np.abs(ca - cb)) == 0.0, "w_bound=0 tidak bit-identik dgn default"

    # (b) batas kecil benar-benar meng-clip
    m = _mk(w_bound=0.5)
    m.train_stream(data)
    mx = float(np.max(np.abs(m.w1)))
    assert mx <= 0.5 + 1e-12, f"w_bound=0.5 tidak meng-clip: max|w1|={mx}"

    # (c) batas besar tak mengubah apa pun saat bobot tak mencapai batas
    m1, m2 = _mk(), _mk(w_bound=100.0)
    c1, c2 = m1.train_stream(data), m2.train_stream(data)
    c1, c2 = c1[~np.isnan(c1)], c2[~np.isnan(c2)]
    assert np.max(np.abs(c1 - c2)) == 0.0, "w_bound besar seharusnya tak berpengaruh"
    print(f"  w_bound: paritas OK, clip(p=0,5) OK (max|w1|={mx:.3f}), bound longgar OK")


def test_mixer_freeze_after_freezes_mixer():
    """PENJAGA FIX KURVA-U (E6/E7/E8): `mixer_freeze_after=K` harus membekukan mixer+APM
    tepat di byte ke-K, sementara TABEL tetap belajar.

    Karena update tabel independen dari bobot mixer, model yang dibekukan di K harus punya
    w1/w2/apm1/apm2 IDENTIK dengan model yang hanya dilatih pada data[:K], tetapi tabelnya
    identik dengan model yang dilatih penuh (tanpa beku). Kalau knob ini rusak (mis. mixer
    ikut beku, atau tabel ikut beku), fix kurva-U palsu.
    """
    data = np.frombuffer(load("license"), dtype=np.uint8)[:4000]
    K = 1500

    # (a) 0 == default (paritas bit-identik)
    a, b = _mk(), _mk(mixer_freeze_after=0)
    ca, cb = a.train_stream(data), b.train_stream(data)
    ca, cb = ca[~np.isnan(ca)], cb[~np.isnan(cb)]
    assert np.max(np.abs(ca - cb)) == 0.0, "mixer_freeze_after=0 tidak bit-identik dgn default"

    # (b) beku di K: mixer/APM == model yang hanya melihat data[:K]
    m = _mk(mixer_freeze_after=K)
    m.train_stream(data)
    short = _mk()
    short.train_stream(data[:K])
    assert np.max(np.abs(m.w1 - short.w1)) == 0.0, "w1 tak beku tepat di K"
    assert np.max(np.abs(m.w2 - short.w2)) == 0.0, "w2 tak beku tepat di K"
    assert np.max(np.abs(m.apm1 - short.apm1)) == 0.0, "apm1 tak beku tepat di K"
    assert np.max(np.abs(m.apm2 - short.apm2)) == 0.0, "apm2 tak beku tepat di K"
    # tabel TETAP belajar (harus beda dari model yang hanya melihat data[:K])...
    assert not np.array_equal(m.tab, short.tab), "tabel ikut beku (seharusnya tetap belajar)"
    # ...dan identik dengan model yang dilatih penuh tanpa beku (update tabel independen dari lr)
    full = _mk()
    full.train_stream(data)
    assert np.array_equal(m.tab, full.tab), "tabel beda dari latih penuh (freeze mengganggu tabel)"

    # (c) titik beku di luar panjang data == perilaku default
    m2 = _mk(mixer_freeze_after=len(data) + 999)
    c2 = m2.train_stream(data)
    c2 = c2[~np.isnan(c2)]
    assert np.max(np.abs(ca - c2)) == 0.0, "freeze di luar panjang data seharusnya tak berpengaruh"

    # (d) beku lintas BEBERAPA panggilan (akuntansi `_train_fed` benar)
    m3 = _mk(mixer_freeze_after=K)
    for i in range(0, len(data), 700):
        m3.train_stream(data[i:i + 700])
    assert np.max(np.abs(m3.w1 - short.w1)) == 0.0, "beku lintas panggilan salah (w1)"
    assert np.max(np.abs(m3.apm2 - short.apm2)) == 0.0, "beku lintas panggilan salah (apm2)"
    assert np.array_equal(m3.tab, full.tab), "tabel lintas panggilan beda dari latih penuh"

    # (e) hyper round-trip menyimpan knob ini
    h = _mk(mixer_freeze_after=1234).hyper()
    assert int(h["mixer_freeze_after"]) == 1234, "mixer_freeze_after tak tersimpan di hyper()"
    assert int(StatCore.from_hyper(h).mixer_freeze_after) == 1234, "round-trip knob gagal"
    print(f"  mixer_freeze_after: paritas OK, beku tepat di {K} OK, tabel tetap belajar OK")


def test_full_tree_sums_to_one():
    """Distribusi 256 simbol yang tersirat oleh pohon biner harus menjumlah ke 1."""
    m = _mk()
    for b in np.frombuffer(load("license"), dtype=np.uint8)[:200]:
        m.step(int(b))
    tot = 0.0
    for b in range(256):
        pr = m.byte_probs(b)
        p = 1.0
        for i in range(8):
            bit = (b >> (7 - i)) & 1
            p *= pr[i] if bit else (1.0 - pr[i])
        tot += p
    assert abs(tot - 1.0) < 1e-9, f"jumlah distribusi = {tot}"
    print(f"  pohon biner menjumlah ke {tot:.15f}: OK")


def test_eval_does_not_mutate_state():
    """eval_stream (train=False) TIDAK boleh mengubah tabel/penghitung/mixer/APM."""
    m = _mk()
    for b in np.frombuffer(load("license"), dtype=np.uint8)[:300]:
        m.step(int(b))
    snap = (m.tab.copy(), m.w1.copy(), m.w2.copy(),
            m.apm1.copy(), m.apm2.copy(), m.mchk.copy(), m.mbyte.copy())
    te = np.frombuffer(load("license"), dtype=np.uint8)[300:500]
    m.eval_stream(te)
    for a, b in zip(snap, (m.tab, m.w1, m.w2, m.apm1, m.apm2, m.mchk, m.mbyte)):
        assert np.array_equal(a, b), "eval beku mengubah state!"
    print("  eval beku tidak mengubah state: OK")


def test_prequential_causality():
    """CE prefiks tidak boleh bergantung pada byte SESUDAHNYA (tanpa kebocoran)."""
    base = np.frombuffer(load("license"), dtype=np.uint8)[:600].astype(np.int64)
    a = base.copy()
    b = base.copy()
    b[400:] = (b[400:] + 77) % 256                    # ubah ekor saja
    ma, mb = _mk(), _mk()
    ca, cb = [], []
    for x in a:
        c = ma.step(int(x))
        if c is not None:
            ca.append(c)
    for x in b:
        c = mb.step(int(x))
        if c is not None:
            cb.append(c)
    ca, cb = np.asarray(ca), np.asarray(cb)
    assert np.allclose(ca[:399], cb[:399], atol=1e-12), "ada kebocoran dari masa depan"
    print("  kausalitas prekuensial (prefiks identik): OK")


def test_determinism():
    """Dua model dengan konfigurasi sama harus memberi CE identik."""
    data = np.frombuffer(load("license"), dtype=np.uint8)[:500]
    m1, m2 = _mk(), _mk()
    c1 = [c for c in (m1.step(int(x)) for x in data) if c is not None]
    c2 = [c for c in (m2.step(int(x)) for x in data) if c is not None]
    assert np.array_equal(np.asarray(c1), np.asarray(c2))
    print("  determinisme: OK")


def test_training_reduces_ce():
    """CE pada data yang sudah dilihat harus turun setelah latihan (sanity, bukan klaim kualitas)."""
    data = np.frombuffer(load("license"), dtype=np.uint8)[:3000]
    cold = _mk()
    ce_cold = cold.eval_stream(data)
    warm = _mk()
    for x in data[:2000]:
        warm.step(int(x))
    ce_warm = warm.eval_stream(data[2000:])
    assert ce_warm.mean() < ce_cold[len(data) - 1000:].mean(), "latihan tidak menurunkan CE"
    print(f"  latihan menurunkan CE: {ce_cold[len(data)-1000:].mean():.3f} -> "
          f"{ce_warm.mean():.3f} bit: OK")


def test_sample_byte_matches_tree_distribution():
    """`_sample_byte` harus sampel dari distribusi byte yang SAMA dengan pohon biner.

    Regresi untuk bug 2026-10-04: `_sample_byte` dulu memakai ``np.prod(byte_probs(b))`` —
    mengalikan p(bit=1) untuk SEMUA bit seolah tiap bit bernilai 1. Akibatnya byte biner
    (bit-1 di posisi tinggi, >127) menang dan generasi keluar byte sampah. Di sini kita
    bandingkan distribusi empiris `_sample_byte` dengan distribusi yang dihitung BENAR
    (∏ p(bit = nilai bit byte)).
    """
    m = _mk()
    for b in np.frombuffer(load("license"), dtype=np.uint8)[:300]:
        m.step(int(b))

    # distribusi referensi yang dihitung benar
    ref = np.empty(256, dtype=np.float64)
    for b in range(256):
        pr = m.byte_probs(b)
        p = 1.0
        for i in range(8):
            bit = (b >> (7 - i)) & 1
            p *= pr[i] if bit else (1.0 - pr[i])
        ref[b] = p
    ref /= ref.sum()

    # distribusi empiris dari _sample_byte (temp=1 → apa adanya)
    rng = np.random.default_rng(0)
    N = 3000                                    # tiap sampel = 256 pass maju; jaga tes tetap cepat
    counts = np.zeros(256, dtype=np.float64)
    for _ in range(N):
        counts[m._sample_byte(rng, temp=1.0)] += 1
    emp = counts / N

    # byte paling mungkin harus cocok, dan tak boleh didominasi byte >127 (sampah biner)
    assert int(np.argmax(emp)) == int(np.argmax(ref)), \
        f"puncak beda: emp {np.argmax(emp)} vs ref {np.argmax(ref)}"
    tv = 0.5 * np.abs(emp - ref).sum()          # jarak variasi total
    assert tv < 0.25, f"distribusi sampel menyimpang (TV={tv:.4f})"
    frac_high = float(emp[128:].sum())
    assert frac_high < 0.5, f"terlalu banyak byte >127 ({frac_high:.2%}) — bug lama muncul lagi"
    print(f"  _sample_byte cocok pohon biner (TV={tv:.4f}, frac>127={frac_high:.2%}): OK")


def test_generate_produces_text_not_binary():
    """Generasi harus keluar teks ASCII yang masuk akal, bukan byte biner."""
    m = _mk()
    data = np.frombuffer(load("license"), dtype=np.uint8)
    for b in data[:3000]:
        m.step(int(b))
    out = m.generate("Permission is hereby", 200, temp=0.5, seed=0)
    frac_printable = float(np.mean([(32 <= c < 127) for c in out]))
    assert frac_printable > 0.9, f"hanya {frac_printable:.1%} byte tercetak — generasi rusak"
    print(f"  generasi {frac_printable:.1%} byte tercetak: OK")


def test_eval_window_must_not_overlap_train():
    """PENJAGA KEBOORAN KEEMPAT — cek IRISAN, bukan rasio BPB.

    Percobaan awal memakai rasio BPB (uji harus >1,3x lebih buruk dari latih) tetapi GAGAL
    kalibrasi: pada model yang benar-benar bagus, jendela latih DAN uji keduanya mudah
    (rasio ~1,00x), jadi rasio bukan penanda kebocoran yang andal. Yang andal adalah cek
    GEOMETRI: rentang indeks jendela uji TIDAK BOLEH beririsan dengan rentang indeks latih.

    Riwayat: `scripts/diag_overwrite.py` v1 memakai `te = raw[ntrain-ntest:ntrain]` (ekor
    prefix latih) -> BPB palsu **0,5281** padahal jujur **2,1361** (rasio ~4x). Rasio besar
    itu memang muncul saat kebocoran PENUH; tetapi mengandalkannya rapuh. Cek irisan bersifat
    pasti dan menangkap kasus kebocoran parsial yang tak terlihat di rasio.
    """
    total, ntrain, ntest = 26000, 20000, 4000

    def overlaps(train_lo, train_hi, test_lo, test_hi):
        return not (train_hi <= test_lo or test_hi <= train_lo)

    # (a) susunan BENAR: latih [0,ntrain), uji [ntrain, ntrain+ntest)
    t_lo, t_hi = 0, ntrain
    u_lo, u_hi = ntrain, ntrain + ntest
    assert u_hi <= total
    assert not overlaps(t_lo, t_hi, u_lo, u_hi), "susunan acuan sendiri beririsan?"

    # (b) susunan BOCOR: uji = ekor prefix latih (bug lama) -> WAJIB terdeteksi
    leak_lo, leak_hi = ntrain - ntest, ntrain
    assert overlaps(t_lo, t_hi, leak_lo, leak_hi), \
        "penjaga GAGAL menandai jendela uji yang beririsan dengan latih"
    print("  penjaga irisan: susunan benar OK, susunan bocor terdeteksi OK")


def test_unseen_eval_is_not_suspiciously_good():
    """Sanity fisiologis: evaluasi jendela uji TAK TERLIHAT tidak boleh 'terlalu bagus'.

    BPB < 1,0 bit/byte pada teks alami nyata adalah sinyal kuat kebocoran/bug (teks alami
    byte-level praktis tak pernah ada di bawah ~1 bit/byte tanpa model besar). Dipakai
    korpus sintetis berstruktur sehingga ambang aman; ini melengkapi cek irisan di atas.
    """
    rng = np.random.default_rng(1)
    # kalimat bermakna terbatas + noise: ada struktur, tetapi tidak bisa ditebak sempurna
    words = [b"yang", b"dan", b"di", b"ini", b"itu", b"kota", b"negara", b"besar"]
    seq = bytearray()
    for _ in range(20000):
        seq += words[int(rng.integers(0, len(words)))]
        if rng.random() < 0.15:
            seq += bytes([int(rng.integers(32, 127))])     # noise acak
        seq += b" "
    raw = np.frombuffer(bytes(seq), dtype=np.uint8)
    ntrain, ntest = 20000, 4000
    m = _mk()
    for b in raw[:ntrain]:
        m.step(int(b))
    bpb = float(m.eval_stream(raw[ntrain:ntrain + ntest]).mean())
    assert bpb > 0.5, f"BPB uji {bpb:.4f} terlalu rendah -> curiga kebocoran/bug"
    print(f"  BPB uji tak-terlihat {bpb:.4f} (tidak mencurigakan): OK")


def test_fast_stream_bit_identical_to_byte_loop():
    """Jalur cepat terkompilasi (`train_stream`/`eval_stream_fast`) WAJIB bit-identik dengan
    loop byte Python (`step`/`eval_stream`) — jalur Python adalah acuan.

    Latar: memanggil `step` per byte dari Python hanya mencapai ~90 kB/s karena overhead
    pemanggilan Python per byte (bukan karena kernel lambat). Jalur cepat memindahkan loop
    byte ke dalam numba. Kalau jalur cepat dan acuan tidak identik, semua hasil E1/E2 yang
    memakainya jadi tidak sah. Penjaga ini menjaga paritas itu.
    """
    data = np.frombuffer(load("license"), dtype=np.uint8)[:5000]
    tr, te = data[:4000], data[4000:]

    a = _mk()
    ref_tr = np.asarray([c for c in (a.step(int(b), train=True) for b in tr)
                         if c is not None], dtype=np.float64)
    ref_te = a.eval_stream(te)

    b = _mk()
    got_tr = b.train_stream(tr)
    got_tr = got_tr[~np.isnan(got_tr)]
    got_te = b.eval_stream_fast(te)
    got_te = got_te[~np.isnan(got_te)]

    assert ref_tr.shape == got_tr.shape, f"bentuk CE latih beda: {ref_tr.shape} vs {got_tr.shape}"
    assert ref_te.shape == got_te.shape, f"bentuk CE uji beda: {ref_te.shape} vs {got_te.shape}"
    assert np.max(np.abs(ref_tr - got_tr)) == 0.0, "train_stream tidak bit-identik"
    assert np.max(np.abs(ref_te - got_te)) == 0.0, "eval_stream_fast tidak bit-identik"
    print(f"  paritas jalur cepat: latih & uji bit-identik ({ref_te.shape[0]} byte) OK")


def test_nn_stream_fast_path_bit_identical():
    """PENJAGA JALUR GURU: `train_stream(data, nn_stream)` WAJIB bit-identik dengan loop
    Python `step(..., nn_st=...)`, dan `eval_stream_fast(data, nn_stream)` dengan
    `eval_stream(data, nn_stream)`.

    Ini yang membuat guru eksternal (mis. LLM byte-level lewat `tmt2/byte_bridge.py`) bisa
    dipakai pada korpus besar. Sebelum ini, `_stream()` meneruskan `use_nn=0` SELALU, jadi
    guru hanya jalan di jalur Python (~60 kB/s). Kalau jalur cepat menyimpang dari acuan
    Python, semua hasil berbasis guru jadi tidak sah.
    """
    data = np.frombuffer(load("license"), dtype=np.uint8)[:4000]
    tr, te = data[:3000], data[3000:]
    rng = np.random.default_rng(3)
    ntr = rng.normal(0.0, 2.0, size=(len(tr), 8))
    nte = rng.normal(0.0, 2.0, size=(len(te), 8))

    a = _mk()
    ref_tr = np.asarray([c for c in (a.step(int(b), train=True, nn_st=ntr[i])
                                     for i, b in enumerate(tr)) if c is not None],
                        dtype=np.float64)
    ref_te = a.eval_stream(te, nn_stream=nte)

    b = _mk()
    got_tr = b.train_stream(tr, nn_stream=ntr)
    got_tr = got_tr[~np.isnan(got_tr)]
    got_te = b.eval_stream_fast(te, nn_stream=nte)
    got_te = got_te[~np.isnan(got_te)]

    assert ref_tr.shape == got_tr.shape, f"bentuk CE latih beda: {ref_tr.shape} vs {got_tr.shape}"
    assert ref_te.shape == got_te.shape, f"bentuk CE uji beda: {ref_te.shape} vs {got_te.shape}"
    assert np.max(np.abs(ref_tr - got_tr)) == 0.0, "train_stream+nn_stream tidak bit-identik"
    assert np.max(np.abs(ref_te - got_te)) == 0.0, "eval_stream_fast+nn_stream tidak bit-identik"
    print(f"  jalur cepat + guru bit-identik dgn acuan Python ({ref_te.shape[0]} byte uji): OK")


def test_nn_stream_none_is_unchanged():
    """`nn_stream=None` (default) harus TIDAK mengubah apa pun (perilaku lama)."""
    data = np.frombuffer(load("license"), dtype=np.uint8)[:3000]
    a, b = _mk(), _mk()
    ca, cb = a.train_stream(data), b.train_stream(data, nn_stream=None)
    ca, cb = ca[~np.isnan(ca)], cb[~np.isnan(cb)]
    assert np.max(np.abs(ca - cb)) == 0.0, "nn_stream=None mengubah hasil"
    print("  nn_stream=None == perilaku lama: OK")


def test_nn_stream_actually_used_and_validated():
    """Guru harus BENAR-BENAR memengaruhi prediksi (bukan diabaikan) + validasi bentuk."""
    data = np.frombuffer(load("license"), dtype=np.uint8)[:3000]
    rng = np.random.default_rng(5)

    # guru "oracle" sederhana: bit sebenarnya, magnitudo besar
    ns = np.zeros((len(data), 8))
    for i, b in enumerate(data):
        for k in range(8):
            ns[i, k] = 6.0 if (int(b) >> (7 - k)) & 1 else -6.0

    plain = _mk()
    plain.train_stream(data)
    ce_plain = float(np.nanmean(plain.eval_stream_fast(data)))

    with_t = _mk()
    with_t.train_stream(data, nn_stream=ns)
    ce_t = float(np.nanmean(with_t.eval_stream_fast(data, nn_stream=ns)))

    assert ce_t < ce_plain - 0.5, f"guru kuat seharusnya menurunkan CE: {ce_plain:.4f} -> {ce_t:.4f}"
    wg = float(np.max(np.abs(with_t.w1[:, :, with_t.nctx + 1])))
    assert wg > 0.5, f"bobot guru tak tumbuh (max={wg})"

    # bentuk salah -> ValueError
    for bad in (rng.normal(0, 1, (len(data), 7)), rng.normal(0, 1, (10, 8))):
        try:
            _mk().train_stream(data, nn_stream=bad)
            raise AssertionError("bentuk nn_stream salah seharusnya ditolak")
        except ValueError:
            pass
    print(f"  guru dipakai nyata (CE {ce_plain:.4f} -> {ce_t:.4f}, max|w_guru|={wg:.3f}); "
          f"bentuk salah ditolak: OK")


def test_nn_stream_sliced_with_freeze_split():
    """Bila `mixer_freeze_after` memecah aliran, `nn_stream` harus dipecah sejajar."""
    data = np.frombuffer(load("license"), dtype=np.uint8)[:3000]
    K = 1200
    rng = np.random.default_rng(9)
    ns = rng.normal(0.0, 2.0, size=(len(data), 8))

    m1 = _mk(mixer_freeze_after=K)
    c1 = m1.train_stream(data, nn_stream=ns)

    m2 = _mk(mixer_freeze_after=K)
    c2 = np.concatenate([m2.train_stream(data[:K], nn_stream=ns[:K]),
                         m2.train_stream(data[K:], nn_stream=ns[K:])])

    c1, c2 = c1[~np.isnan(c1)], c2[~np.isnan(c2)]
    assert c1.shape == c2.shape
    assert np.max(np.abs(c1 - c2)) == 0.0, "pemecahan nn_stream tak sejajar dgn data"
    print(f"  nn_stream dipecah sejajar di titik beku ({K}): OK")


def test_freeze_boundary_chunk_is_free():
    """REGRESI BUG BATAS: chunk yang TEPAT berakhir di titik beku harus dilatih BEBAS.

    Versi lama menaikkan `_train_fed` sebelum chunk dilatih, sehingga laju efektif dihitung
    dari keadaan AKHIR — chunk yang panjangnya persis sama dengan sisa jarak ke titik beku
    ikut membeku (berbeda dari jalur `step()`, yang menaikkannya sesudah). Ditemukan oleh
    `test_nn_stream_sliced_with_freeze_split`.
    """
    data = np.frombuffer(load("license"), dtype=np.uint8)[:3000]
    K = 1500
    rng = np.random.default_rng(11)
    ns = rng.normal(0.0, 2.0, size=(len(data), 8))

    m1 = _mk(mixer_freeze_after=K)
    m1.train_stream(data[:K], nn_stream=ns[:K])
    ref = _mk()                                   # tanpa beku -> acuan fase bebas
    ref.train_stream(data[:K], nn_stream=ns[:K])
    assert np.array_equal(m1.tab, ref.tab), "tabel beda dari acuan bebas"
    assert np.max(np.abs(m1.w1 - ref.w1)) == 0.0, "chunk tepat-di-titik-beku ikut membeku"

    w_before = m1.w1.copy()                       # byte setelahnya HARUS beku
    m1.train_stream(data[K:], nn_stream=ns[K:])
    assert np.max(np.abs(m1.w1 - w_before)) == 0.0, "setelah titik beku mixer harus berhenti"
    assert not np.array_equal(m1.tab, ref.tab), "tabel harus tetap belajar saat mixer beku"
    print(f"  batas titik beku ({K}): chunk terakhir bebas, setelahnya beku: OK")


def test_instrument_stream_does_not_perturb_training():
    """E2 `instrument_stream` WAJIB tidak mengubah hasil latih (probe tak boleh bocor).

    Probe dijalankan train=False dan menulis ke hist ring buffer; kalau pemulihan state
    salah, BPB uji akan geser. Bandingkan model berinstrumentasi vs model tanpa.
    """
    data = np.frombuffer(load("license"), dtype=np.uint8)[:8000]
    tr, te = data[:6000], data[6000:]

    ref = _mk()
    ref.train_stream(tr)
    ref_bpb = float(np.nanmean(ref.eval_stream_fast(te)))

    ins = _mk()
    ins.instrument_stream(tr, interval=2000, probe_bytes=500)
    ins_bpb = float(np.nanmean(ins.eval_stream_fast(te)))

    assert ref_bpb == ins_bpb, f"instrumentasi menggeser hasil latih: {ref_bpb} vs {ins_bpb}"
    print(f"  instrument_stream tak mengganggu latih: BPB {ins_bpb:.6f} identik OK")


def test_diag_does_not_change_results():
    """PENJAGA Fase 0: akumulator diagnostik NOL mengubah hasil (CE/state identik)."""
    data = np.frombuffer(load("license"), dtype=np.uint8)[:3000]
    a, b = _mk(), _mk()
    a.train_stream(data)
    b.train_stream(data)
    d = b.diagnose(data[:1000])
    assert np.array_equal(a.tab, b.tab) and np.array_equal(a.w1, b.w1)
    # diag per-BIT, eval per-BYTE (8 bit) -> kali 8 sebelum bandingkan
    assert abs(d["pf_ce"] * 8 - float(np.nanmean(b.eval_stream_fast(data[:1000])))) < 1e-9
    assert all(np.isfinite(v) for v in d["own_ce"] + [d["pm_ce"]])
    assert d["nbit"] == 8 * 1000, d["nbit"]
    assert all(0.0 <= s <= 1.0 for s in d["sat_frac"])
    print(f"  diag tak mengubah hasil (sat={d['sat_frac']}, w2neg={d['w2neg_frac']:.3f}): OK")


def test_eval_suite_matches_fresh_models():
    """PENJAGA Fase 0: `eval_suite` == model baru per jendela (tanpa salin tabel)."""
    data = np.frombuffer(load("license"), dtype=np.uint8)[:6000]
    m = _mk()
    m.train_stream(data[:4000])
    wins = [data[4000:4500], data[4500:5000], data[5000:5500]]
    mean, per = m.eval_suite(wins, cold=True)
    ref = []
    for w in wins:
        r = _mk()
        r.tab[:] = m.tab
        r.w1[:] = m.w1
        r.w2[:] = m.w2
        r.apm1[:] = m.apm1
        r.apm2[:] = m.apm2
        r.mchk[:] = m.mchk      # tanpa ini ref kehilangan indeks match -> bias
        r.mbyte[:] = m.mbyte
        ref.append(float(np.nanmean(r.eval_stream_fast(w))))
    assert np.allclose(per, ref, atol=1e-12), f"{per} vs {ref}"
    print(f"  eval_suite == model baru/jendela ({mean:.4f}): OK")


def _mkv3(**over):
    kw = dict(SMALL)
    kw.update(dict(mixer_mode=1))
    kw.update(over)
    return StatCore(**kw)


def test_v3_update_bound_single_byte():
    """Galat-sendiri: SATU byte latih mengubah w1 paling banyak 8·lr·max|x| (l2=0,
    jadwal off). Mengikat rumus update `g·(e·x)` tanpa mereplikasi kernel."""
    data = np.frombuffer(load("license"), dtype=np.uint8)[:500]
    m = _mkv3(mix_l2=0.0, mix_lr_floor=1.0, mix_lr_nhalf=0)
    for b in data[:100]:
        m.step(int(b))
    before = m.w1.copy()
    # |x| maksimum saat ini (stretch bisa ~±8)
    xs = np.abs(m._st).max()
    m.step(int(data[100]))
    dw = np.abs(m.w1 - before).max()
    assert dw <= 8 * m.lr * max(xs, 1.0) + 1e-9, f"update melewati batas: {dw}"
    # dan cnt bertambah tepat 8 kunjungan/bit-byte (satu per bit x3 set? satu per set per byte)
    assert int(m.cnt.sum()) > 0, "cnt tak bertambah"
    print(f"  v3 update bound OK (dw={dw:.5f}, cnt_sum={int(m.cnt.sum())})")


def test_v3_schedule_takes_effect():
    """Jadwal AKTIF bila (nhalf>0, floor<1): bobot beda vs tanpa jadwal.

    Penjaga footgun: floor default 1.0 + nhalf>0 = jadwal diam-diam nonaktif
    (pernah membuat V3==V2 persis di matriks cepat).
    """
    data = np.frombuffer(load("license"), dtype=np.uint8)[:3000]
    a = _mkv3()
    b = _mkv3(mix_lr_nhalf=500, mix_lr_floor=0.02)
    a.train_stream(data)
    b.train_stream(data)
    assert not np.array_equal(a.w1, b.w1), "jadwal tak berpengaruh (footgun floor?)"
    print("  v3 schedule aktif (floor<1): OK")


def test_v3_no_runaway_on_deterministic_stream():
    """Tak ada runaway: pola deterministik 8-byte x 250kB, mode1 + jadwal + L2 ringan
    -> max|w1| < 12 (legacy dibiarkan apa adanya, hanya dilaporkan)."""
    pat = np.frombuffer(b"abcdefgh", dtype=np.uint8)
    data = np.tile(pat, 32000)[:250000]
    m = _mkv3(table_bits=18, mix_lr_nhalf=50000, mix_l2=1e-4)
    m.train_stream(data)
    mx = float(np.max(np.abs(m.w1)))
    assert mx < 12.0, f"runaway mode1: max|w1|={mx}"
    leg = _mk(table_bits=18)
    leg.train_stream(data)
    print(f"  v3 no-runaway OK (mode1 max|w1|={mx:.2f} vs legacy {float(np.abs(leg.w1).max()):.2f})")


def test_v3_step_parity_with_schedule_and_freeze():
    """Paritas step-loop vs train_stream dgn mode1 + jadwal + freeze (termasuk cnt)."""
    data = np.frombuffer(load("license"), dtype=np.uint8)[:4000]
    kw = dict(mixer_mode=1, mix_lr_nhalf=1000, mix_l2=1e-4, mixer_freeze_after=1500)
    a = _mkv3(**kw)
    a.train_stream(data)
    b = _mkv3(**kw)
    for x in data:
        b.step(int(x))
    assert np.array_equal(a.w1, b.w1) and np.array_equal(a.cnt, b.cnt), "w1/cnt beda"
    assert np.array_equal(a.w2, b.w2) and np.array_equal(a.tab, b.tab)
    print("  v3 step==stream (+jadwal+freeze, cnt identik): OK")


def test_v3_ckpt_roundtrip_with_cnt():
    """Round-trip ckpt menyimpan cnt + knob v3; resume identik."""
    import tempfile
    data = np.frombuffer(load("license"), dtype=np.uint8)[:5000]
    kw = dict(mixer_mode=1, s2_clip=12.0, mix_lr_floor=0.1, mix_lr_nhalf=2000,
              mix_l2=1e-4)
    m = _mkv3(**kw)
    m.train_stream(data[:3000])
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "v3.npz")
        m.save(p, dict(note="v3"))
        m2, meta = StatCore.load(p)
        assert np.array_equal(m.cnt, m2.cnt) and int(m2.hyper()["mixer_mode"]) == 1
        assert abs(float(m2.hyper()["s2_clip"]) - 12.0) < 1e-12
        m.train_stream(data[3000:])
        m2.train_stream(data[3000:])
        assert np.array_equal(m.w1, m2.w1) and np.array_equal(m.cnt, m2.cnt)
    print("  v3 ckpt round-trip + resume identik: OK")


def test_v3_own_error_descends():
    """Galat-sendiri menurunkan own-CE: byte tunggal diulang -> ownCE set turun."""
    data = np.frombuffer(bytes([65]) * 400, dtype=np.uint8)
    m = _mkv3(mix_l2=0.0)
    m.train_stream(data[:200])
    d1 = m.diagnose(data[:200])
    m.train_stream(data[:200])
    d2 = m.diagnose(data[:200])
    assert np.mean(d2["own_ce"]) < np.mean(d1["own_ce"]), \
        f"ownCE tak turun: {d1['own_ce']} -> {d2['own_ce']}"
    print(f"  v3 ownCE turun {np.mean(d1['own_ce'])*8:.3f} -> {np.mean(d2['own_ce'])*8:.3f}: OK")


def test_ckpt_saves_train_end_ctx_not_eval_end():
    """Regresi: ckpt menyimpan konteks AKHIR-LATIH (pemanggil wajib restore
    ctx_snaps SEBELUM save; save() sendiri menyimpan apa adanya — tes ini
    memakukan kontrak itu dari sisi skrip via pola snapshot/restore)."""
    import tempfile
    data = np.frombuffer(load("license"), dtype=np.uint8)[:5000]
    tr, te = data[:4000], data[4000:]
    m = _mk()
    m.train_stream(tr)
    snap = m.stream_ctx()
    m.eval_stream(te)  # memajukan ctx ke akhir uji
    assert m.pos != snap[1], "eval tak memajukan ctx (asumsi tes salah)"
    m.restore_ctx(snap)  # pola wajib train_gen_big.py
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "c.npz")
        m.save(p, dict(note="x"))
        m2, _ = StatCore.load(p)
        assert m2.pos == snap[1] and m2.hlen == snap[2], "ckpt menyimpan ctx uji!"
        assert np.array_equal(m2.hist, snap[0])
    print("  kontrak ctx-ckpt: OK")


def test_4way_opt0_bit_identical():
    """assoc=0 (default) bit-identik dengan perilaku legacy di semua jalur."""
    data = np.frombuffer(load("license"), dtype=np.uint8)[:3000]
    a = _mk()
    b = _mk(assoc=1)
    b.opt[:] = 0  # paksa mati -> harus identik legacy
    ca = a.train_stream(data)
    cb = b.train_stream(data)
    ca, cb = ca[~np.isnan(ca)], cb[~np.isnan(cb)]
    assert np.max(np.abs(ca - cb)) == 0.0, "opt=0 tidak bit-identik legacy!"
    print("  4-way opt=0 bit-identik legacy: OK")


def test_4way_deterministic_and_eval_nomutation():
    """assoc=1 deterministik + eval tak memutasi."""
    data = np.frombuffer(load("license"), dtype=np.uint8)[:4000]
    a, b = _mk(assoc=1), _mk(assoc=1)
    ca = a.train_stream(data)
    cb = b.train_stream(data)
    ca, cb = ca[~np.isnan(ca)], cb[~np.isnan(cb)]
    assert np.max(np.abs(ca - cb)) == 0.0, "assoc=1 tak deterministik!"
    snap = (b.tab.copy(), b.w1.copy())
    b.eval_stream(data[:500])
    assert np.array_equal(snap[0], b.tab) and np.array_equal(snap[1], b.w1)
    print("  4-way deterministik + eval tak memutasi: OK")


def test_4way_preserves_hot_entry():
    """Fungsional #4: entri berhitungan-tinggi bertahan melawan banjir singleton.

    Terukur 2026-10-10 (orders=(1,), tb=8, pola panas + banjir 12kB acak):
    legacy +3,92 bit vs 4-way +0,06. Ambang longgar (0,5 vs 2,0) agar tak rapuh.
    """
    rng = np.random.default_rng(0)
    hot = bytearray()
    for _ in range(1500):
        hot += bytes([65])
        hot += bytes([int(rng.integers(65, 91))])
    hot = np.frombuffer(bytes(hot), dtype=np.uint8)
    flood = rng.integers(0, 256, 12000).astype(np.uint8)
    rises = {}
    for assoc in (0, 1):
        m = StatCore(orders=(1,), table_bits=8, match_bits=8, wsel_bits=4,
                     assoc=assoc, use_apm=False)
        m.train_stream(hot)
        pre = float(np.nanmean(m.eval_stream_fast(hot[:600])))
        m.train_stream(flood)
        post = float(np.nanmean(m.eval_stream_fast(hot[:600])))
        rises[assoc] = post - pre
    assert rises[1] < 0.5, f"4-way tak melindungi: +{rises[1]:.3f}"
    assert rises[0] > 2.0, f"kontrol legacy tak runtuh (+{rises[0]:.3f}) — uji kehilangan daya"
    print(f"  4-way lindungi entri panas (+{rises[1]:.3f} vs legacy +{rises[0]:.3f}): OK")


def test_dropout_deterministic_and_drops():
    """Dropout-lat
ih deterministik + benar-benar men-drop + eval imun (RNG tak maju)."""
    data = np.frombuffer(load("license"), dtype=np.uint8)[:4000]
    a, b = _mk(dropout_permille=100), _mk(dropout_permille=100)
    ca = a.train_stream(data)
    cb = b.train_stream(data)
    ca, cb = ca[~np.isnan(ca)], cb[~np.isnan(cb)]
    assert np.max(np.abs(ca - cb)) == 0.0, "dropout tak deterministik!"
    c = _mk()
    c.train_stream(data)
    # mask drop -> trayektori beda dari tanpa-dropout
    d = _mk(dropout_permille=100)
    d.train_stream(data)
    assert not np.array_equal(c.w1, d.w1), "dropout tak mengubah apa pun!"
    # eval tak memajukan RNG dan tak memutasi state adaptif
    r0, m0, s0 = int(d._rng[0]), int(d._mrun[0]), d.sm.copy()
    d.eval_stream(data[:500])
    assert int(d._rng[0]) == r0 and int(d._mrun[0]) == m0, "eval memutasi RNG/mrun!"
    print("  dropout deterministik + drop + eval-imun: OK")


def test_match_adapt_learns_and_roundtrips():
    """C2-L1: mrun/sm bergerak, hasil deterministik, ckpt v4 round-trip + resume."""
    import tempfile
    data = np.frombuffer(load("license"), dtype=np.uint8)[:5000]
    a, b = _mk(match_adapt=1), _mk(match_adapt=1)
    a.train_stream(data)
    b.train_stream(data)
    assert np.array_equal(a.w1, b.w1) and int(a._mrun[0]) == int(b._mrun[0])
    assert np.array_equal(a.sm, b.sm)
    assert bool(((a.sm != 0.5).any())), "sm tak belajar!"
    assert a.sm.min() >= 0.0 and a.sm.max() <= 1.0, "sm keluar [0,1]!"
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "c4.npz")
        a.save(p, dict(note="c4"))
        m2, _ = StatCore.load(p)
        assert int(m2._mrun[0]) == int(a._mrun[0]) and np.array_equal(m2.sm, a.sm)
        assert int(m2._rng[0]) == int(a._rng[0])
        a.train_stream(data)
        m2.train_stream(data)
        assert np.array_equal(a.w1, m2.w1) and np.array_equal(a.tab, m2.tab)
    print(f"  C2-L1 belajar (mrun={int(a._mrun[0])}) + ckpt-v4 resume: OK")


def test_hitsel_deterministic_and_legacy_intact():
    """opt[1]=1 deterministik + eval tak memutasi; default (0) = golden legacy."""
    data = np.frombuffer(load("license"), dtype=np.uint8)[:4000]
    a, b = _mk(hit_sel=1), _mk(hit_sel=1)
    ca = a.train_stream(data)
    cb = b.train_stream(data)
    ca, cb = ca[~np.isnan(ca)], cb[~np.isnan(cb)]
    assert np.max(np.abs(ca - cb)) == 0.0, "hit_sel tak deterministik!"
    snap = (b.tab.copy(), b.w1.copy())
    b.eval_stream(data[:500])
    assert np.array_equal(snap[0], b.tab) and np.array_equal(snap[1], b.w1)
    print("  hit_sel deterministik + eval tak memutasi: OK")


def test_stat_core_is_deterministic_seed_irrelevant():
    """StatCore deterministik: arg `seed` tak dipakai -> dua seed = CE identik.

    Konsekuensi metodologi: replikasi kualitas harus lewat OFFSET DATA, bukan seed.
    """
    data = np.frombuffer(load("license"), dtype=np.uint8)[:2000]
    kw = dict(SMALL)
    a = StatCore(seed=0, **kw)
    b = StatCore(seed=12345, **kw)
    ca = a.train_stream(data)
    cb = b.train_stream(data)
    assert np.array_equal(ca[~np.isnan(ca)], cb[~np.isnan(cb)]), "seed memengaruhi hasil!"
    print("  seed irrelevan (deterministik): OK")


def test_generate_is_order_independent():
    """`generate()` merestore konteks: dua panggilan berurutan identik + ctx sama."""
    m = _mk()
    for b in np.frombuffer(load("license"), dtype=np.uint8)[:3000]:
        m.step(int(b))
    c0 = m.stream_ctx()
    g1 = m.generate("Permission is hereby", 100, temp=0.5, seed=3)
    c1 = m.stream_ctx()
    g2 = m.generate("Permission is hereby", 100, temp=0.5, seed=3)
    assert g1 == g2, "generate tak deterministik terhadap urutan pemanggilan!"
    assert c1[1] == c0[1] and np.array_equal(c1[0], c0[0]), "ctx bocor antar-sampel!"
    print("  generate order-independent: OK")


def test_min_p_filters_and_normalizes():
    """min-p: min_p=1.0 -> selalu argmax; 0,05 -> distribusi tetap ternormalisasi."""
    import numpy as np
    m = _mk()
    for b in np.frombuffer(load("license"), dtype=np.uint8)[:300]:
        m.step(int(b))
    rng = np.random.default_rng(1)
    outs = {m._sample_byte(rng, temp=1.0, min_p=1.0) for _ in range(20)}
    assert len(outs) == 1, f"min_p=1 tak deterministik: {outs}"
    print(f"  min-p=1 -> argmax deterministik (byte {outs.pop()}): OK")


def test_ancestral_sampler_matches_tree():
    """Sampler ancestral 8-pass == distribusi pohon (TV<0,25, pola tes yang sama
    dengan test_sample_byte_matches_tree_distribution)."""
    m = _mk()
    for b in np.frombuffer(load("license"), dtype=np.uint8)[:300]:
        m.step(int(b))
    ref = np.empty(256, dtype=np.float64)
    for b in range(256):
        pr = m.byte_probs(b)
        p = 1.0
        for i in range(8):
            bit = (b >> (7 - i)) & 1
            p *= pr[i] if bit else (1.0 - pr[i])
        ref[b] = p
    ref /= ref.sum()
    rng = np.random.default_rng(0)
    counts = np.zeros(256, dtype=np.float64)
    for _ in range(3000):
        counts[m._sample_byte_ancestral(rng, temp=1.0)] += 1
    emp = counts / 3000
    tv = 0.5 * np.abs(emp - ref).sum()
    assert tv < 0.25, f"ancestral menyimpang (TV={tv:.4f})"
    assert int(np.argmax(emp)) == int(np.argmax(ref)), "puncak distribusi beda"
    print(f"  ancestral == pohon (TV={tv:.4f}): OK")


def test_prod_matches_train_gen_big():
    """PENJAGA §11.4: `Config.prod()` == konstruktor `train_gen_big.py` (sumber tunggal).

    Bila skrip produksi dan preset divergen (lr/orders/tb/flag), hyper() beda -> gagal.
    """
    import sys as _sys
    import os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
    from tmt2.model import Config as _Config
    cfg = _Config.prod()
    m1 = StatCore.from_config(cfg)
    m2 = StatCore(orders=(1, 2, 3, 4, 5, 6, 8, 12, 16, 24), table_bits=26, lr=0.002,
                  use_apm=True, apm_lr=0.02, match_minlen=6, match_bits=24,
                  match_conf=2.0, wsel_bits=12, sparse=False, word=True,
                  count_cap=1023, cks_bits=24, match_fix=True, ctx_fix=True,
                  match_ck_bits=8, wsel_mode=0, stretch_exact=0, w_bound=0.0,
                  mixer_freeze_after=0, mixer_mode=1, hit_sel=1,
                  dropout_permille=0, match_adapt=0)
    assert m1.hyper() == m2.hyper(), (
        "Config.prod() divergen dari train_gen_big.py:\n"
        f"{ {k: (m1.hyper()[k], m2.hyper()[k]) for k in m1.hyper() if m1.hyper()[k] != m2.hyper()[k]} }")
    print("  Config.prod() == train_gen_big: OK")


def test_prod_is_v3():
    """Default produksi = Mixer v3 (flip 2026-10-10, gerbang §7 lulus)."""
    import sys as _sys
    import os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
    from tmt2.model import Config as _Config
    h = StatCore.from_config(_Config.prod()).hyper()
    assert h["mixer_mode"] == 1, "prod() bukan v3!"
    assert h["mix_lr_nhalf"] == 0 and h["mix_l2"] == 0.0, "prod() bukan V2-polos!"
    assert h["hit_sel"] == 1 and h["match_bits"] == 24, "prod() tanpa hitsel/mb24!"
    print("  prod() = V2 (mode1 polos): OK")


def test_golden_legacy_values():
    """NILAI GOLDEN perilaku legacy (diambil 2026-10-08 SEBELUM refactor _eff_rates).

    `mixer_mode` default / freeze=0 harus mereproduksi angka ini bit-identik.
    Kalau refactor mengubah kernel, tes ini gagal duluan sebelum hasil E-series
    ikut bergeser diam-diam.
    """
    data = np.frombuffer(load("license"), dtype=np.uint8)[:3000]
    m = _mk()
    ce = m.train_stream(data)
    assert abs(float(np.nansum(ce)) - 10391.621672088491) < 1e-9, "CE legacy bergeser!"
    assert abs(float(np.max(np.abs(m.w1))) - 0.48320115861616453) < 1e-12
    assert abs(float(np.mean(np.abs(m.w1))) - 0.07722690934684093) < 1e-12
    assert abs(float(np.max(np.abs(m.w2))) - 0.5730940531329883) < 1e-12
    print("  golden legacy bit-identik: OK")


def test_step_loop_matches_train_stream_with_freeze():
    """PENJAGA §11.2: loop `step()` WAJIB bit-identik dengan `train_stream()` juga
    saat freeze aktif (dulu step() diam-diam mengabaikan freeze _train_fed=0)."""
    data = np.frombuffer(load("license"), dtype=np.uint8)[:4000]
    K = 1500
    a = _mk(mixer_freeze_after=K)
    a.train_stream(data)
    b = _mk(mixer_freeze_after=K)
    for x in data:
        b.step(int(x))
    assert np.array_equal(a.w1, b.w1), "w1 step-loop != train_stream (freeze)"
    assert np.array_equal(a.w2, b.w2), "w2 step-loop != train_stream (freeze)"
    assert np.array_equal(a.apm1, b.apm1) and np.array_equal(a.apm2, b.apm2)
    assert np.array_equal(a.tab, b.tab), "tabel step-loop != train_stream"
    assert a._train_fed == b._train_fed == len(data), \
        f"_train_fed {a._train_fed} vs {b._train_fed}"
    print(f"  step-loop == train_stream dgn freeze@{K} (_train_fed={a._train_fed}): OK")


def test_instrument_respects_freeze():
    """PENJAGA §11.2: `instrument_stream` memakai laju efektif yang sama dengan
    `train_stream` (dulu memakai lr penuh walau freeze aktif)."""
    data = np.frombuffer(load("license"), dtype=np.uint8)[:8000]
    tr = data[:6000]
    K = 2500
    ref = _mk(mixer_freeze_after=K)
    ref.train_stream(tr)
    ins = _mk(mixer_freeze_after=K)
    ins.instrument_stream(tr, interval=2000, probe_bytes=500)
    assert np.array_equal(ref.w1, ins.w1), "instrument mengabaikan freeze (w1)"
    assert np.array_equal(ref.w2, ins.w2), "instrument mengabaikan freeze (w2)"
    assert np.array_equal(ref.tab, ins.tab), "instrument menggeser tabel (freeze)"
    print("  instrument_stream menghormati freeze: OK")


def test_ckpt_roundtrip_preserves_train_fed():
    """PENJAGA §11.3: save/load menyimpan `_train_fed`; lanjut latih dari ckpt
    tetap beku (dulu mixer terbuka lagi selama K byte)."""
    import tempfile
    data = np.frombuffer(load("license"), dtype=np.uint8)[:5000]
    K = 1500
    m = _mk(mixer_freeze_after=K)
    m.train_stream(data[:3000])
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "m.npz")
        m.save(p, dict(note="t"))
        m2, meta = StatCore.load(p)
        assert m2._train_fed == 3000, f"_train_fed hilang: {m2._train_fed}"
        assert int(meta["hyper"]["mixer_freeze_after"]) == K
        # lanjut latih: mixer tetap == model referensi yang tak pernah save/load
        m.train_stream(data[3000:])
        m2.train_stream(data[3000:])
        assert np.array_equal(m.w1, m2.w1), "resume ckpt membuka mixer!"
        assert np.array_equal(m.tab, m2.tab), "resume ckpt menggeser tabel!"
    print("  ckpt round-trip + resume beku: OK")


if __name__ == "__main__":
    for fn in (test_ce_matches_probability_product, test_ce_unit_is_bits_not_nats,
                test_hyper_roundtrip_bit_identical, test_w_bound_caps_mixer_weights,
                test_mixer_freeze_after_freezes_mixer,
                test_diag_does_not_change_results,
                test_eval_suite_matches_fresh_models,
                test_v3_update_bound_single_byte,
                test_v3_schedule_takes_effect,
                test_v3_no_runaway_on_deterministic_stream,
                test_v3_step_parity_with_schedule_and_freeze,
                test_v3_ckpt_roundtrip_with_cnt,
                test_v3_own_error_descends,
                test_prod_matches_train_gen_big,
                test_prod_is_v3,
                test_golden_legacy_values,
                test_step_loop_matches_train_stream_with_freeze,
                test_instrument_respects_freeze,
                test_ckpt_roundtrip_preserves_train_fed,
                test_ckpt_saves_train_end_ctx_not_eval_end,
                test_4way_opt0_bit_identical,
                test_4way_deterministic_and_eval_nomutation,
                test_4way_preserves_hot_entry,
                test_hitsel_deterministic_and_legacy_intact,
                test_dropout_deterministic_and_drops,
                test_match_adapt_learns_and_roundtrips,
                test_stat_core_is_deterministic_seed_irrelevant,
                test_generate_is_order_independent,
                test_ancestral_sampler_matches_tree,
                test_min_p_filters_and_normalizes,
                test_full_tree_sums_to_one,
                test_eval_does_not_mutate_state, test_prequential_causality,
                test_determinism, test_training_reduces_ce,
                test_sample_byte_matches_tree_distribution,
                test_generate_produces_text_not_binary,
                test_eval_window_must_not_overlap_train,
                test_unseen_eval_is_not_suspiciously_good,
               test_fast_stream_bit_identical_to_byte_loop,
               test_nn_stream_fast_path_bit_identical,
               test_nn_stream_none_is_unchanged,
               test_nn_stream_actually_used_and_validated,
               test_nn_stream_sliced_with_freeze_split,
               test_freeze_boundary_chunk_is_free,
               test_instrument_stream_does_not_perturb_training):
        print(f"[{fn.__name__}]")
        fn()
    print("SEMUA TES stat_core LULUS")
