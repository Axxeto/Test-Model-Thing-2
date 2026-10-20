"""Tes loader korpus, termasuk penjaga bug konversi byte.

Bug yang dijaga: ``bytes(rng.integers(0, V, n))`` memakai BUFFER numpy mentah,
bukan nilainya. Untuk int64 hasilnya ``8*n`` byte dengan hanya tiap byte ke-8
berisi nilai. Ini pernah membuat semua probe "byte acak" salah (7/8 nol) dan
memunculkan pola periodik palsu. ``random_bytes`` mengonversi eksplisit.
"""
import numpy as np
import pytest
from tmt2.corpus import random_bytes, license_bytes, periodic, load, sanity_check


def test_random_bytes_length_and_range():
    """Panjang harus n (bukan 8n), dan semua nilai < vocab."""
    b = random_bytes(512, 64, 0)
    assert len(b) == 512, f"panjang {len(b)} != 512 (buffer numpy bocor?)"
    vals = np.frombuffer(b, dtype=np.uint8)
    assert vals.max() < 64
    assert len(np.unique(vals)) > 50, "nilai tidak cukup beragam -> mungkin buffer mentah"


def test_bytes_of_numpy_is_a_trap():
    """Dokumentasi langsung: bytes(array) != nilai; random_bytes benar."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 64, 512)
    naive = bytes(arr)
    assert len(naive) == 4096, "asumsi bug (int64 -> 4096 byte)"
    # hanya tiap byte ke-8 yang berisi nilai; sisanya nol -> >= 7*512 = 3584 nol
    zeros = sum(1 for i, x in enumerate(naive) if x == 0)
    assert zeros >= 3584, zeros
    # implementasi kanonik TIDAK punya cacat ini
    good = random_bytes(512, 64, 0)
    assert len(good) == 512
    assert sum(1 for x in good if x == 0) < 40


def test_random_bytes_deterministic():
    assert random_bytes(64, 32, 5) == random_bytes(64, 32, 5)
    assert random_bytes(64, 32, 5) != random_bytes(64, 32, 6)


def test_license_and_periodic_are_bytes():
    lic = license_bytes()
    assert isinstance(lic, bytes) and len(lic) > 1000
    per = periodic()
    assert isinstance(per, bytes) and len(per) > 10


def test_load_returns_bytes():
    for kind in ("license", "periodic", "synthetic"):
        b = load(kind, n=200)
        assert isinstance(b, bytes) and len(b) == 200


def test_sanity_check_catches_bug_data():
    """Penjaga: data hasil `bytes(int64)` harus DITOLAK oleh sanity_check."""
    rng = np.random.default_rng(0)
    bad = bytes(rng.integers(0, 64, 512))           # 4096 byte, 7/8 nol
    with pytest.raises(ValueError):
        sanity_check(bad, vocab=64, name="bad", expect_len=512)
    ok = random_bytes(512, 64, 0)
    info = sanity_check(ok, vocab=64, name="ok", expect_len=512)
    assert info["n"] == 512 and not info["problems"]
    assert 5.5 < info["H0"] < 6.0                    # ~log2 64


def test_sanity_check_rejects_bad_range():
    with pytest.raises(ValueError):
        sanity_check(b"\xff\xfe\xfd\x00", vocab=16, name="range")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name} ok")
