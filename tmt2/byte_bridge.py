"""Jembatan TOKEN -> BYTE -> BIT: memakai LLM token-level (mis. Qwen3.5-4B) sebagai guru
**byte-level** untuk TMT2 — tanpa melatih ulang LLM.

Masalah: Qwen3.5-4B memprediksi TOKEN (BPE, vocab 248.044, decoder ByteLevel); TMT2 butuh
log-odds per BIT dari byte berikutnya (`nn_st`, 8 nilai per byte). Tidak ada saklar "mode byte"
pada LLM token-level, dan Qwen3.5 **tidak punya token byte-fallback `<0xXX>`** (dicek: 0 buah),
jadi memaksa token byte juga bukan opsi.

Solusi (eksak, tanpa latih ulang): model token-level mendefinisikan p(token | konteks).
Byte berikutnya = byte pertama token yang terpilih, maka

    p(byte_j = b | b_1..b_{j-1}) =
        (massa token yang konsisten dgn prefiks & byte ke-j = b)
        / (massa token yang konsisten dgn prefiks & masih berlanjut)

Setelah prefiks habis (tak ada token kandidat yang bisa berlanjut) -> panggil LLM lagi. Jadi LLM
dijalankan **sekali per TOKEN** (~2-3 byte), bukan sekali per byte.

Dari distribusi byte 256-nilai, log-odds per bit dihitung dengan menjumlahkan bit MSB-first —
persis format `StatCore.step(..., nn_st=...)`.
"""
import json
import math

import numpy as np

# ---------------------------------------------------------------- ByteLevel (GPT-2) mapping


def gpt2_byte_decoder():
    """Kebalikan peta `bytes_to_unicode` GPT-2: char -> byte (0..255)."""
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("\u00a1"), ord("\u00ac") + 1))
          + list(range(ord("\u00ae"), ord("\u00ff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def token_to_bytes(tok_str, dec):
    """Byte mentah dari token ByteLevel (vocab disimpan ter-encode, bukan byte mentah)."""
    out = bytearray()
    for ch in tok_str:
        b = dec.get(ch)
        if b is None:                       # bukan char ByteLevel -> perlakukan sebagai UTF-8
            out.extend(ch.encode("utf-8"))
        else:
            out.append(b)
    return bytes(out)


def load_token_bytes(path, include_added=False):
    """(M, lens, ids): M (V,L) uint8 = byte tiap token (padded), lens (V,), ids (V,) = id asli.

    Token khusus/added (mis. `<|im_start|>`) DIKELUARKAN secara default — ia bukan teks, jadi
    tak boleh masuk distribusi byte.
    """
    d = json.load(open(path, encoding="utf-8"))
    vocab = d["model"]["vocab"]
    dec = gpt2_byte_decoder()
    special = {a["content"] for a in d.get("added_tokens", [])}
    items = [(t, i) for t, i in vocab.items() if include_added or t not in special]
    seqs = [token_to_bytes(t, dec) for t, _ in items]
    L = max(len(s) for s in seqs)
    M = np.zeros((len(seqs), L), dtype=np.uint8)
    lens = np.zeros(len(seqs), dtype=np.int64)
    for r, s in enumerate(seqs):
        M[r, :len(s)] = np.frombuffer(s, dtype=np.uint8)
        lens[r] = len(s)
    return M, lens, np.array([i for _, i in items], dtype=np.int64)


# ---------------------------------------------------------------- byte -> bit


def bits_logodds(p256, byte):
    """Distribusi byte (256,) + byte SEBENARNYA -> 8 log-odds p(bit=1 | bit-bit sebelumnya).

    `byte` menentukan prefiks: bit ke-i dikondisikan pada bit-bit sebenarnya di atasnya —
    persis urutan dekode TMT2 (MSB-first). Mengembalikan 8 nilai = format `nn_st`.
    """
    p = np.asarray(p256, dtype=np.float64)
    s = p.sum()
    if not np.isfinite(s) or s <= 0:
        return np.zeros(8)
    p = p / s
    beta = np.arange(256)
    out = np.zeros(8)
    for i in range(8):
        if i == 0:
            m = np.ones(256, dtype=bool)
        else:
            m = (beta >> (8 - i)) == (int(byte) >> (8 - i))
        b = beta[m]
        w = p[m]
        tot = float(w.sum())
        if tot <= 1e-300:
            out[i] = 0.0
            continue
        p1 = float(w[((b >> (7 - i)) & 1) == 1].sum())
        q = min(max(p1 / tot, 1e-9), 1.0 - 1e-9)
        out[i] = math.log(q / (1.0 - q))
    return out


# ---------------------------------------------------------------- jembatan


class ByteBridge:
    """Turunkan arus byte (dan log-odds bit) dari distribusi token LLM."""

    def __init__(self, tokenizer_json):
        self.M, self.lens, self.ids = load_token_bytes(tokenizer_json)
        self.V, self.L = self.M.shape
        self._p = None
        self._alive = None
        self._j = 0

    # -- siklus hidup: satu distribusi token dipakai sampai token selesai
    def start_token(self, p_token):
        """`p_token`: (V,) peluang atas BARIS vocab (bukan id mentah)."""
        p = np.asarray(p_token, dtype=np.float64)
        if p.shape[0] != self.V:
            raise ValueError(f"p_token harus panjang {self.V}, dapat {p.shape[0]}")
        self._p = p
        self._alive = p > 0
        self._j = 0

    def byte_distribution(self):
        """p(byte berikutnya) dari token yang masih konsisten DAN masih berlanjut."""
        cont = self._alive & (self.lens > self._j)
        if not cont.any():
            return None
        p = np.bincount(self.M[cont, self._j], weights=self._p[cont], minlength=256)
        s = p.sum()
        return (p / s) if s > 0 else None

    def advance(self, byte):
        self._alive &= self.M[:, self._j] == byte
        self._j += 1

    def can_continue(self):
        return bool((self._alive & (self.lens > self._j)).any())

    def bits_for_token(self, p_token, actual_bytes):
        """(len(actual_bytes), 8) log-odds untuk token yang byte-nya `actual_bytes`."""
        self.start_token(p_token)
        rows = []
        for b in actual_bytes:
            p256 = self.byte_distribution()
            if p256 is None:
                break
            rows.append(bits_logodds(p256, b))
            self.advance(b)
            if not self.can_continue():
                break
        return np.array(rows) if rows else np.zeros((0, 8))
