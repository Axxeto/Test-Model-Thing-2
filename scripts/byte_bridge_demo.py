"""Demo + validasi jembatan token->byte->bit dengan tokenizer Qwen3.5-4B ASLI.

Menjawab: "agar LLM seperti Qwen memakai byte, bukan token" — tanpa melatih ulang LLM.

Yang ditunjukkan:
1. Teks -> token (Qwen) -> distribusi byte per byte (lewat marginalisasi trie).
2. Uji rekonstruksi: argmax distribusi byte harus mengembalikan teks PERSIS.
3. Amortisasi: LLM dipanggil sekali per TOKEN, bukan per byte (laporkan byte/token).
4. Uji Monte-Carlo: sampel token -> detokenisasi; distribusi byte empiris harus cocok
   dengan yang diturunkan.
5. Arus stretch siap disuapkan ke TMT2 (`nn_st`).

Jalan:  python scripts/byte_bridge_demo.py --text data/indonesian_1m_pure.txt --n 400
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np
from tokenizers import Tokenizer

from tmt2.byte_bridge import ByteBridge, gpt2_byte_decoder, token_to_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="data/tokenizers/qwen35_4b_tokenizer.json")
    ap.add_argument("--text", default="data/indonesian_1m_pure.txt")
    ap.add_argument("--n", type=int, default=400, help="jumlah token yang diproses")
    ap.add_argument("--smooth", type=float, default=0.02, help="peluang 'LLM tidak yakin'")
    a = ap.parse_args()

    tok = Tokenizer.from_file(a.tokenizer)
    br = ByteBridge(a.tokenizer)
    row_of_id = {int(i): r for r, i in enumerate(br.ids)}
    print(f"[bridge] vocab baris={br.V} (maks panjang token={br.L} byte), "
          f"tokenizer vocab={tok.get_vocab_size()}")

    raw = open(a.text, "rb").read()
    # ambil potongan teks yang valid UTF-8
    txt = raw[:20000].decode("utf-8", errors="ignore")
    enc = tok.encode(txt)
    ids = enc.ids[:a.n]
    print(f"[bridge] teks {len(txt)} byte -> {len(enc.ids)} token "
          f"({len(txt)/max(len(enc.ids),1):.2f} byte/token)")

    dec = gpt2_byte_decoder()
    rng = np.random.default_rng(0)

    # ---- bangun arus guru: satu distribusi token per posisi (di sini: "LLM sempurna" + smoothing)
    stream, n_llm_calls, ok_byte = [], 0, 0
    text_bytes = bytearray()
    for k, tid in enumerate(ids):
        row = row_of_id.get(int(tid))
        if row is None:                       # token khusus -> lewati
            continue
        tbytes = bytes(br.M[row, :br.lens[row]])
        # "distribusi LLM": massa utama di token benar + smoothing seragam
        p = np.full(br.V, a.smooth / br.V)
        p[row] += 1.0 - a.smooth
        p /= p.sum()
        rows = br.bits_for_token(p, tbytes)
        n_llm_calls += 1
        stream.append(rows)
        text_bytes.extend(tbytes)
        # cek rekonstruksi argmax untuk token ini
        br.start_token(p)
        for j, b in enumerate(tbytes):
            p256 = br.byte_distribution()
            if p256 is None or int(np.argmax(p256)) != b:
                break
            br.advance(b)
        else:
            ok_byte += 1
    S = np.vstack(stream)
    print(f"[bridge] LLM dipanggil {n_llm_calls}x untuk {len(text_bytes)} byte "
          f"({len(text_bytes)/max(n_llm_calls,1):.2f} byte per panggilan)")
    print(f"[bridge] arus stretch: {S.shape} (8 log-odds per byte)")
    print(f"[bridge] rekonstruksi argmax: {ok_byte}/{n_llm_calls} token persis")

    # ---- cek jangkauan & sign
    fin = np.isfinite(S).all()
    print(f"[bridge] semua nilai finite: {fin}; rentang [{S.min():.2f}, {S.max():.2f}]")

    # ---- uji Monte-Carlo distribusi byte di satu batas token
    br.start_token(np.ones(br.V) / br.V)      # LLM "seragam" -> distribusi byte = frekuensi token
    p256 = br.byte_distribution()
    samp = rng.choice(br.V, size=200000, p=np.ones(br.V) / br.V)
    first = br.M[samp, 0]
    emp = np.bincount(first, minlength=256) / len(samp)
    tv = 0.5 * np.abs(emp - p256).sum()
    print(f"[bridge] uji Monte-Carlo dist. byte pertama: TV={tv:.5f} (harus ~0)")

    # ---- buktikan arus ini dipakai TMT2 lewat JALUR CEPAT (train_stream + nn_stream)
    from tmt2.stat_core import StatCore
    data = np.frombuffer(bytes(text_bytes), dtype=np.uint8)
    assert len(S) == len(data), f"arus guru ({len(S)}) tak sejajar dgn byte ({len(data)})"

    def mk():
        return StatCore(orders=(0, 1, 2, 3, 4, 5, 6), table_bits=16, wsel_bits=8,
                        match_fix=True, ctx_fix=True, use_apm=True, mixer_mode=1)

    # best-of-2 tiap jalur: panggilan pertama menyertakan waktu kompilasi numba (cold start)
    def t_fast():
        m = mk()
        t = time.perf_counter()
        m.train_stream(data, nn_stream=S)
        return time.perf_counter() - t, m

    def t_ref():
        m = mk()
        t = time.perf_counter()
        for i, b in enumerate(data):
            m.step(int(b), train=True, nn_st=S[i])
        return time.perf_counter() - t, m

    _t, m_fast = t_fast()                       # warm-up (kompilasi)
    tf = min(t_fast()[0], t_fast()[0])
    _t, m_ref = t_ref()
    tr_ = min(t_ref()[0], t_ref()[0])
    par = bool(np.array_equal(m_fast.w1, m_ref.w1) and np.array_equal(m_fast.tab, m_ref.tab))
    print(f"[bridge] jalur cepat + guru: {len(data)/tf:.0f} B/s vs loop Python "
          f"{len(data)/tr_:.0f} B/s -> {tr_/tf:.1f}x  (paritas bit-identik={par})")
    print(f"[bridge] TMT2 memakai guru: max|w_guru|="
          f"{np.max(np.abs(m_fast.w1[:, :, m_fast.nctx + 1])):.3f}")


if __name__ == "__main__":
    main()
