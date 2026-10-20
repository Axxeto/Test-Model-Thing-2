"""Pemuatan korpus lintas-platform untuk skrip TMT2.

Skrip asli membaca ``/usr/share/common-licenses/{GPL-3,Apache-2.0}`` (khas Debian/Ubuntu).
Di Windows (atau mesin tanpa paket itu) berkas tersebut tidak ada, jadi loader ini mencoba
berurutan:

  1. ``/usr/share/common-licenses/<nama>``            -> perilaku asli di Linux
  2. ``<akar proyek>/data/licenses/<nama>``           -> korpus yang dibundel di repo ini
  3. sumber Python/numpy sebagai teks nyata cadangan  -> selalu tersedia

Semua fungsi mengembalikan ``bytes``; pemotongan ke ``n`` byte dilakukan oleh pemanggil.
"""
import os
import numpy as np

_LICENSES = ("GPL-3", "Apache-2.0")                      # urutan sama seperti skrip asli
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PERIODIC = b"the quick brown fox jumps over the lazy dog. "


def _try_read(path):
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError:
        return None


def license_bytes():
    """Gabungan GPL-3 + Apache-2.0 dari lokasi pertama yang tersedia."""
    out = b""
    for name in _LICENSES:
        for path in (f"/usr/share/common-licenses/{name}",
                     os.path.join(_ROOT, "data", "licenses", name)):
            data = _try_read(path)
            if data:
                out += data
                break
    if len(out) >= 5000:
        return out
    # cadangan terakhir: teks nyata yang pasti ada bersama Python
    import inspect
    return (inspect.getsource(np.linalg) * 3).encode()


def periodic():
    """Teks periodik panjang (baseline konteks murni, tanpa statistik bahasa)."""
    return _PERIODIC * 4000                               # ~180 KB


_SENT = (b"the cat sat on the mat. ", b"a dog runs in the park. ", b"birds fly over the lake. ")


def synthetic(n=None, seed=0):
    """3 variasi kalimat dipilih acak -> byte berikutnya hanya bisa diprediksi dengan konteks."""
    rng = np.random.default_rng(seed)
    out = bytearray()
    n = n or 200000
    while len(out) < n:
        out += _SENT[int(rng.integers(0, len(_SENT)))]
    return bytes(out[:n])


def random_bytes(n=512, vocab=64, seed=0):
    """``n`` byte acak i.i.d. dari nilai ``[0, vocab)``, satu byte per elemen.

    PENTING. Jangan pakai ``bytes(rng.integers(0, vocab, n))``: ``bytes()`` pada array
    numpy memakai BUFFER mentah, bukan nilai. Untuk int64 itu menghasilkan ``8*n`` byte
    dengan hanya tiap byte ke-8 yang berisi nilai (sisanya nol). Bug ini pernah membuat
    probe "512 byte acak" sebenarnya berisi 4096 byte dengan 7/8 nol, dan menciptakan
    pola periodik palsu (pola "periode-8"). Selalu konversi eksplisit lewat uint8/tobytes.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, vocab, n).astype(np.uint8).tobytes()


def sanity_check(data, vocab=None, name="data", expect_len=None, raise_on_fail=True):
    """Cetak & periksa statistik dasar aliran byte SEBELUM menafsirkan probe apa pun.

    Mengembalikan dict; bila ``raise_on_fail`` dan ada pelanggaran berat -> ``ValueError``.
    Ini penjaga terhadap kelas bug "bytes(ndarray)" (7/8 nol) dan salah panjang.

    Pelanggaran berat:
      * ``expect_len`` diberikan tetapi panjang aktual berbeda.
      * ``vocab`` diberikan tetapi ada nilai ``>= vocab``.
      * fraksi nilai yang sama dengan nilai paling sering > ``max_mode_frac`` (default 0,9):
        sinyal data hampir konstan (mis. padding nol).
    Info (tidak menaikkan): ``H0`` empiris vs ``log2(vocab)``.
    """
    b = np.frombuffer(data, dtype=np.uint8)
    uniq, cnt = np.unique(b, return_counts=True)
    mode_frac = float(cnt.max() / len(b)) if len(b) else 0.0
    H0 = float(-np.sum((cnt / len(b)) * np.log2(cnt / len(b) + 1e-12))) if len(b) else 0.0
    info = dict(name=name, n=len(b), uniq=len(uniq), vmin=int(b.min()) if len(b) else -1,
                vmax=int(b.max()) if len(b) else -1, mode_frac=mode_frac, H0=H0)
    print(f"[sanity:{name}] n={info['n']} uniq={info['uniq']} range=[{info['vmin']},{info['vmax']}] "
          f"mode_frac={mode_frac:.3f} H0={H0:.4f} bit"
          + (f" (log2 V={np.log2(vocab):.4f})" if vocab else ""))
    problems = []
    if expect_len is not None and len(b) != expect_len:
        problems.append(f"panjang {len(b)} != {expect_len}")
    if vocab is not None and len(b) and int(b.max()) >= vocab:
        problems.append(f"nilai {int(b.max())} >= vocab {vocab}")
    if mode_frac > 0.9:
        problems.append(f"mode_frac={mode_frac:.3f} > 0.9 (data hampir konstan?)")
    if problems:
        msg = f"[sanity:{name}] GAGAL: " + "; ".join(problems)
        print(msg)
        if raise_on_fail:
            raise ValueError(msg)
    info["problems"] = problems
    return info


def wikitext():
    """WikiText (validasi) bila tersedia di data/wikitext_valid.txt."""
    p = os.path.join(_ROOT, "data", "wikitext_valid.txt")
    data = _try_read(p)
    if data is None:
        raise FileNotFoundError(
            "data/wikitext_valid.txt tidak ditemukan; taruh berkas WikiText di sana")
    return data


def copy_task(L=32, n=20000, V=4, seed=0, perm=False, cycle=False):
    """Probe memori: blok acak panjang L diulang SEKALI, lalu blok baru yang segar.

    n-gram dan pembobotan tidak bisa menghafal isinya; satu-satunya cara menang adalah menyimpan
    blok di state lalu membacanya kembali setelah lag L. Mengembalikan (ids, mask_paruh_kedua).
    Batas ideal: paruh kedua 0 bit, paruh pertama log2(V).

    perm=True (butuh L <= V) -> SETIAP kunci 1-token UNIK di dalam blok. Tanpa ini, pada V kecil
    blok bisa berisi simbol berulang sehingga "hafalan" dan "prediksi satu langkah" tak terbedakan
    dan probe menjadi ambigu. Lihat catatan di memory_probe.py.

    cycle=False (default) -> tiap blok adalah permutasi BARU; hafalan panjang L wajib.
    cycle=True            -> SATU permutasi dipakai berulang tiap blok (perhatikan: ini kembali
                             membuat kunci satu-token ambigu lewat batas blok; hanya untuk kontrol).
    """
    rng = np.random.default_rng(seed)
    if perm and L > V:
        raise ValueError(f"perm=True butuh L <= V, dapat L={L} V={V}")
    out = []
    fixed = rng.permutation(V)[:L].tolist() if perm else None
    while len(out) < n:
        if perm:
            blk = fixed if cycle else rng.permutation(V)[:L].tolist()
        else:
            blk = rng.integers(0, V, L).tolist()
        out += blk + blk
    ids = np.asarray(out[:n], dtype=np.int64)
    second = (np.arange(n) % (2 * L)) >= L
    return ids, second


def tinystories(which="train"):
    """TinyStoriesV2-GPT4 (2,23 GB train / 22,5 MB valid). Cari di lokasi yang umum dipakai.

    Kaggle: unduh ke /kaggle/working lewat curl. Ukuran penuh train ~2,23 GB -> pemakaian
    harus dibatasi dengan ``n`` (lihat catatan throughput di README/KAGGLE.md).
    """
    names = (f"TinyStoriesV2-GPT4-{which}.txt", f"tinystories_{which}.txt")
    roots = (os.path.join(_ROOT, "data"), "/kaggle/working", "/kaggle/input", os.getcwd(), "/tmp")
    for name in names:
        for r in roots:
            data = _try_read(os.path.join(r, name))
            if data:
                return data
    raise FileNotFoundError(
        f"{names[0]} tidak ditemukan. Cari di: {', '.join(roots)}")


def load(kind, n=None):
    """kind: 'license' | 'periodic' | 'synthetic' | 'wikitext' | 'tinystories'
    | 'tinystories_valid' | path berkas apa pun.
    Mengembalikan korpus penuh (atau n byte pertama) sebagai bytes."""
    if kind == "periodic":
        raw = periodic()
    elif kind == "synthetic":
        raw = synthetic(n)
    elif kind == "wikitext":
        raw = wikitext()
    elif kind == "tinystories":
        raw = tinystories("train")
    elif kind in ("tinystories_valid", "tinystories-valid"):
        raw = tinystories("valid")
    elif os.path.isfile(kind):                      # jalur berkas langsung
        raw = _try_read(kind)
        if raw is None:
            raise FileNotFoundError(kind)
    else:
        raw = license_bytes()
    return raw if n is None else raw[:n]


# ---------------------------------------------------------------- sumber Markov (probe konteks)
# DUA generator berbeda hidup di repo ini; lantai entropinya TIDAK sama. Selalu hitung lantai
# dari tabel yang benar-benar dipakai, jangan mengasumsikan eps*log2(V).
#
#   1. deterministik  : table[ctx] -> satu simbol.      Lantai = eps*log2(V).  (markov_probe.py)
#   2. kolom-acak     : tab[ctx, randint(0,V)].         Lantai = E[H(dist baris)]. (ablation,
#                       yaitu: dengan peluang 1-eps keluarkan entri acak dari V entri baris itu,
#                       sehingga distribusi bersyarat tersebar -> lantai jauh lebih tinggi.
#
# Pelajaran (masuk LAPORAN §10): sebelum memakai angka lantai sebagai tolok ukur, HITUNG dari
# tabel generator yang sebenarnya. Lantai 0,30 bit HANYA sah untuk generator deterministik;
# untuk generator kolom-acak (V=8, k=2) lantainya ~2,25 bit.

def markov_det_stream(n, k, V=8, eps=0.1, seed=0):
    """Generator Markov DETERMINISTIK: T[konteks k] -> satu simbol, noise eps.

    Mengembalikan ``(ids int64, floor_bit_per_simbol)`` dengan ``floor = eps*log2(V)``.
    """
    rng = np.random.default_rng(seed)
    nCtx = V ** k
    table = rng.integers(0, V, nCtx)
    out = np.empty(n, dtype=np.int64)
    ctx = 0
    for i in range(n):
        s = int(rng.integers(0, V)) if rng.random() < eps else int(table[ctx])
        out[i] = s
        ctx = (ctx * V + s) % nCtx
    return out, float(eps * np.log2(V))


def markov_randcol_stream(n, k, V=8, eps=0.1, seed=0):
    """Generator Markov KOLOM-ACAK (dipakai `ablation_mixer.py` & probe lama).

    Penerus = ``tab[ctx, randint(0, V)]``: dengan peluang ``1-eps`` keluarkan simbol acak dari
    MULTISET V entri baris konteks itu; dengan peluang ``eps`` simbol seragam.

    Mengembalikan ``(ids int64, floor_bit_per_simbol)`` dengan lantai = entropi bersyarat ratarata
    (merata-rata atas konteks, bobot seragam). Untuk V=8, k=2 nilai tipikal ~2,25 bit.
    """
    rng = np.random.default_rng(seed)
    nCtx = V ** k
    tab = rng.integers(0, V, size=(nCtx, V))
    out = np.empty(n, dtype=np.int64)
    ctx = 0
    for i in range(n):
        e = 1.0 if rng.random() < eps else 0.0
        if e:
            s = int(rng.integers(0, V))
        else:
            s = int(tab[ctx, int(rng.integers(0, V))])
        out[i] = s
        ctx = (ctx * V + s) % nCtx
    floor = markov_randcol_floor(tab)
    return out, float(floor)


def markov_randcol_floor(tab, eps=0.1):
    """Entropi bersyarat E_ctx[H(P(simbol|ctx))] untuk tabel kolom-acak, dalam bit/simbol.

    ``tab`` berbentuk (nCtx, V). Distribusi bersyarat baris = frekuensi simbol di baris itu,
    dicampur dengan seragam dengan bobot ``eps``.
    """
    nCtx = tab.shape[0]
    tot = 0.0
    for idx in range(nCtx):
        cnt = np.bincount(tab[idx], minlength=int(tab.max()) + 1).astype(np.float64)
        p = cnt / cnt.sum()
        q = (1.0 - eps) * p + eps / len(p)              # campur noise seragam
        q = q[q > 0]
        tot += float(-np.sum(q * np.log2(q)))
    return tot / nCtx
