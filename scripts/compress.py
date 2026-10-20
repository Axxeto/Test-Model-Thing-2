"""TMT2 sebagai KOMPRESOR lossless: arithmetic coding biner di atas p(bit) model.

Protokol (prekuensial, identik di kedua sisi):
  enkoder: probs = byte_probs(b) [beku] -> aritmetik-encode 8 bit -> step(b)
  dekoder: probs = byte_probs(.) [beku] -> aritmetik-decode 8 bit -> step(b)
Kedua sisi berevolusi identik -> roundtrip bit-eksak. Header: panjang asli (8B)
+ byte pertama mentah (model butuh 1B riwayat). Ukuran ≈ CE + overhead aritmetik.

Jalan:
  python scripts/compress.py --n 1000000            # 1MB: kompres + bandingkan
  python scripts/compress.py --n 1000000 --no_demo  # hanya roundtrip + rasio
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from tmt2.console import ensure_utf8
from tmt2.stat_core import StatCore

ensure_utf8()

MASK32 = 0xFFFFFFFF
HALF = 0x80000000
QUARTER = 0x40000000
THREEQ = 0xC0000000
PREC = 1 << 16


class BitWriter:
    def __init__(self):
        self.out = bytearray()
        self.acc = 0
        self.nbits = 0

    def write(self, bit):
        self.acc = (self.acc << 1) | (bit & 1)
        self.nbits += 1
        if self.nbits == 8:
            self.out.append(self.acc)
            self.acc = 0
            self.nbits = 0

    def flush(self):
        if self.nbits:
            self.out.append(self.acc << (8 - self.nbits))
        return bytes(self.out)


class BitReader:
    def __init__(self, buf):
        self.buf = buf
        self.pos = 0
        self.acc = 0
        self.nbits = 0

    def read(self):
        if self.nbits == 0:
            self.acc = self.buf[self.pos] if self.pos < len(self.buf) else 0
            self.pos += 1
            self.nbits = 8
        self.nbits -= 1
        return (self.acc >> self.nbits) & 1


class ArithEncoder:
    def __init__(self):
        self.lo, self.hi = 0, MASK32
        self.follow = 0
        self.w = BitWriter()

    def _emit(self, bit):
        self.w.write(bit)
        while self.follow:
            self.w.write(bit ^ 1)
            self.follow -= 1

    def encode_bit(self, bit, p1):
        """p1 = P(bit=1) float (0,1). Interval [lo,hi] dibagi proporsional."""
        f = max(1, min(PREC - 1, int(p1 * PREC)))
        span = self.hi - self.lo + 1
        mid = self.lo + (span * (PREC - f) // PREC) - 1  # batas: 0 di [lo,mid]
        if bit == 0:
            self.hi = mid
        else:
            self.lo = mid + 1
        while True:
            if self.hi < HALF:
                self._emit(0)
                self.lo = (self.lo << 1) & MASK32
                self.hi = ((self.hi << 1) | 1) & MASK32
            elif self.lo >= HALF:
                self._emit(1)
                self.lo = ((self.lo - HALF) << 1) & MASK32
                self.hi = (((self.hi - HALF) << 1) | 1) & MASK32
            elif self.lo >= QUARTER and self.hi < THREEQ:
                self.follow += 1
                self.lo = ((self.lo - QUARTER) << 1) & MASK32
                self.hi = (((self.hi - QUARTER) << 1) | 1) & MASK32
            else:
                break

    def finish(self):
        self.follow += 1
        if self.lo < QUARTER:
            self._emit(0)
        else:
            self._emit(1)
        return self.w.flush()


class ArithDecoder:
    def __init__(self, buf):
        self.r = BitReader(buf)
        self.lo, self.hi = 0, MASK32
        self.code = 0
        for _ in range(32):
            self.code = ((self.code << 1) | self.r.read()) & MASK32

    def decode_bit(self, p1):
        f = max(1, min(PREC - 1, int(p1 * PREC)))
        span = self.hi - self.lo + 1
        mid = self.lo + (span * (PREC - f) // PREC) - 1
        if self.code <= mid:
            bit = 0
            self.hi = mid
        else:
            bit = 1
            self.lo = mid + 1
        while True:
            if self.hi < HALF:
                pass
            elif self.lo >= HALF:
                self.lo -= HALF
                self.hi -= HALF
                self.code -= HALF
            elif self.lo >= QUARTER and self.hi < THREEQ:
                self.lo -= QUARTER
                self.hi -= QUARTER
                self.code -= QUARTER
            else:
                break
            self.lo = (self.lo << 1) & MASK32
            self.hi = ((self.hi << 1) | 1) & MASK32
            self.code = ((self.code << 1) | self.r.read()) & MASK32
        return bit


def make_model(**over):
    kw = dict(orders=(1, 2, 3, 4, 5, 6, 8, 12, 16, 24), table_bits=25,
              lr=0.002, use_apm=True, apm_lr=0.02, match_minlen=6,
              match_bits=24, match_conf=2.0, wsel_bits=12, sparse=False,
              word=True, count_cap=1023, cks_bits=24, match_fix=True,
              ctx_fix=True, mixer_mode=1, hit_sel=1)
    kw.update(over)
    return StatCore(**kw)


def compress(data, table_bits=25):
    """Kembalikan paket bytes. Model mulai dari NOL (defensif, deterministik)."""
    import struct
    data = bytes(data)
    m = make_model(table_bits=table_bits)
    enc = ArithEncoder()
    out_first = data[:1]
    m.step(data[0], train=True)
    for b in data[1:]:
        pr = m.byte_probs(b)  # beku: 8x p(bit=1) di jalur byte ini
        for i in range(8):
            enc.encode_bit((b >> (7 - i)) & 1, float(np.clip(pr[i], 1e-6, 1 - 1e-6)))
        m.step(int(b), train=True)
    body = enc.finish()
    return struct.pack("<Q", len(data)) + out_first + body


def decompress(pkg, table_bits=25):
    """Kebalikan bit-eksak: lintasan model identik (probabilitas + update sama)."""
    import struct
    (n,) = struct.unpack("<Q", pkg[:8])
    first = pkg[8:9]
    body = pkg[9:]
    m = make_model(table_bits=table_bits)
    dec = ArithDecoder(body)
    out = bytearray(first)
    m.step(first[0], train=True)
    for _ in range(1, n):
        # bit 0..7: probs dihitung dari prefiks yang SUDAH di-decode
        b = 0
        for i in range(8):
            # dummy dengan prefiks benar: p_i hanya bergantung bit <i
            pr = m.byte_probs((b << (8 - i)) & 255)
            bit = dec.decode_bit(float(np.clip(pr[i], 1e-6, 1 - 1e-6)))
            b = (b << 1) | bit
        out.append(b)
        m.step(int(b), train=True)
    return bytes(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/indonesian_big.bin")
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--table_bits", type=int, default=25)
    ap.add_argument("--no_demo", action="store_true")
    a = ap.parse_args()
    with open(a.corpus, "rb") as f:
        raw = f.read(a.n + 1000)[:a.n]

    t0 = time.perf_counter()
    pkg = compress(raw, a.table_bits)
    dt = time.perf_counter() - t0
    back = decompress(pkg, a.table_bits)
    assert back == raw, "ROUNDTRIP GAGAL!"
    bpb = len(pkg) * 8.0 / len(raw)
    print(f"TMT2 {len(raw)}B -> {len(pkg)}B = {bpb:.4f} bit/byte "
          f"({len(raw)/dt:.0f} B/s encode), roundtrip OK", flush=True)
    if not a.no_demo:
        import bz2
        import lzma
        import zlib
        for nm, fn in (("bz2-9", lambda d: bz2.compress(d, 9)),
                       ("lzma-6", lambda d: lzma.compress(d, preset=6)),
                       ("zlib-9", lambda d: zlib.compress(d, 9))):
            c = fn(raw)
            print(f"  {nm}: {len(c)}B = {len(c)*8.0/len(raw):.4f} bit/byte", flush=True)


if __name__ == "__main__":
    main()
