"""Tes jembatan token->byte->bit (`tmt2/byte_bridge.py`).

Yang dijaga:
1. Distribusi byte hasil marginalisasi == brute force atas token (eksak).
2. Jumlah peluang byte = 1.
3. `bits_logodds` konsisten: bit=1 -> log-odds positif, bit=0 -> negatif; dan p(bit) cocok
   dengan marginalisasi bit dari distribusi byte.
4. Prefiks bit mengikuti byte SEBENARNYA (kondisional bertingkat, MSB-first).
5. Selesai-token: berhenti saat tak ada kandidat yang bisa berlanjut (amortisasi LLM per token).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from tmt2.byte_bridge import ByteBridge, bits_logodds, gpt2_byte_decoder   # noqa: E402


def gpt2_byte_encoder():
    bs = (list(range(ord("!"), ord("~") + 1)) + list(range(ord("\u00a1"), ord("\u00ac") + 1))
          + list(range(ord("\u00ae"), ord("\u00ff") + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def _mini_tokenizer(tmpdir, toks):
    """tokenizer.json minimal bergaya Qwen (BPE + decoder ByteLevel)."""
    enc = gpt2_byte_encoder()
    vocab = {"".join(enc[b] for b in t): i for i, t in enumerate(toks)}
    d = dict(version="1.0", added_tokens=[], normalizer=None,
             pre_tokenizer={"type": "ByteLevel"}, post_processor=None,
             decoder={"type": "ByteLevel"},
             model={"type": "BPE", "vocab": vocab, "merges": []})
    p = os.path.join(tmpdir, "tok.json")
    json.dump(d, open(p, "w", encoding="utf-8"))
    return p


TOKS = [b"H", b"e", b"l", b"o", b"llo", b"He", b" ", b"dunia", b"d", b"uni", b"a"]


def test_byte_marginal_matches_brute_force(tmp_path):
    p = _mini_tokenizer(str(tmp_path), TOKS)
    br = ByteBridge(p)
    p_tok = np.array([0.05, 0.05, 0.1, 0.1, 0.4, 0.2, 0.02, 0.04, 0.01, 0.02, 0.01])
    p_tok = p_tok / p_tok.sum()
    br.start_token(p_tok)
    got = br.byte_distribution()
    # brute force: byte pertama tiap token
    want = np.zeros(256)
    for t, w in zip(TOKS, p_tok):
        want[t[0]] += w
    want = want / want.sum()
    assert np.allclose(got, want, atol=1e-12), "marginalisasi byte != brute force"
    assert abs(got.sum() - 1.0) < 1e-12, "peluang byte tidak berjumlah 1"
    print("  marginal byte == brute force, jumlah = 1: OK")


def test_bit_marginal_consistent(tmp_path):
    p = _mini_tokenizer(str(tmp_path), TOKS)
    br = ByteBridge(p)
    p_tok = np.zeros(len(TOKS))
    p_tok[TOKS.index(b"llo")] = 1.0                      # satu token pasti: 'l','l','o'
    br.start_token(p_tok)
    p256 = br.byte_distribution()
    assert p256[ord("l")] > 0.999, "byte pertama harus 'l'"
    st = bits_logodds(p256, ord("l"))
    # 'l' = 0b01101100 -> bit0=0,bit1=1,bit2=1,bit3=0,bit4=1,bit5=1,bit6=0,bit7=0
    bits = [(ord("l") >> (7 - i)) & 1 for i in range(8)]
    for i, b in enumerate(bits):
        assert (st[i] > 3.0) if b == 1 else (st[i] < -3.0), f"bit {i} (={b}) log-odds {st[i]}"
    print(f"  bits_logodds cocok utk 'l' ({bits}): OK")


def test_bit_conditional_follows_actual_prefix():
    """p(bit_i | bit-bit di atasnya) harus memakai prefiks byte SEBENARNYA."""
    p = np.zeros(256)
    p[0b01101100] = 0.5      # 'l'  -> b0=0, b1=1
    p[0b00110000] = 0.5      # '0'  -> b0=0, b1=0  (sama di bit0, beda di bit1)
    st = bits_logodds(p, 0b01101100)
    # bit0: kedua kandidat bit0=0 -> p1=0 -> sangat negatif
    assert st[0] < -3.0, "bit0 seharusnya yakin 0"
    # bit1 dikondisikan pada bit0=0: 'l' bit1=1, '0' bit1=0 -> p1=0,5 -> log-odds ~0
    assert abs(st[1]) < 1e-6, f"bit1 seharusnya netral (0,5), dapat {st[1]}"
    print("  prefiks bit mengikuti byte sebenarnya: OK")


def test_stops_when_no_continuation(tmp_path):
    p = _mini_tokenizer(str(tmp_path), TOKS)
    br = ByteBridge(p)
    p_tok = np.zeros(len(TOKS))
    p_tok[TOKS.index(b"o")] = 1.0                        # token 1 byte
    rows = br.bits_for_token(p_tok, b"o")
    assert rows.shape == (1, 8), f"token 1 byte harus berhenti setelah 1 byte, dapat {rows.shape}"
    print("  berhenti saat token selesai (1 byte): OK")


def test_multibyte_token_stream(tmp_path):
    p = _mini_tokenizer(str(tmp_path), TOKS)
    br = ByteBridge(p)
    p_tok = np.zeros(len(TOKS))
    p_tok[TOKS.index(b"dunia")] = 1.0
    rows = br.bits_for_token(p_tok, b"dunia")
    assert rows.shape == (5, 8), f"harus 5 byte, dapat {rows.shape}"
    # setiap baris: argmax bit harus sama dengan byte sebenarnya
    for j, b in enumerate(b"dunia"):
        sgn = np.sign(rows[j])
        bits = np.array([1.0 if (b >> (7 - i)) & 1 else -1.0 for i in range(8)])
        assert np.array_equal(sgn, bits), f"byte {j} ({chr(b)}): tanda bit tak cocok"
    print("  token 5-byte -> 5 baris, tanda bit benar semua: OK")


def test_gpt2_decoder_roundtrip():
    dec = gpt2_byte_decoder()
    assert len(dec) == 256
    assert sorted(dec.values()) == list(range(256))
    assert dec["!"] == ord("!") and dec["\u0100"] == 0
    print("  peta ByteLevel GPT-2: 256 entri, roundtrip OK")
