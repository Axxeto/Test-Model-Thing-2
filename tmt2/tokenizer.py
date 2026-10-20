"""BPE subword tokenizer untuk TMT2 (pembungkus pustaka ``tokenizers``).

Model TMT2 memperlakukan id token sebagai "byte" internal, jadi tidak ada logika model yang
berubah: hanya ``Config.vocab``. Modul ini juga mengembalikan PANJANG BYTE tiap token supaya
BPB tetap bisa dihitung eksak dan sebanding dengan baseline byte:

    BPB = (jumlah bit lintas token) / (jumlah byte asli)

  from tmt2 import tokenizer as tk
  tok = tk.get(raw_bytes, 4096, "data/bpe_wikitext_4096.json")
  ids, byte_len, canon = tk.encode(tok, raw_bytes)
"""
import os
import numpy as np

DEFAULT_VOCAB = 4096


def canon_bytes(data_bytes):
    """Dekode lalu encode ulang: menjamin panjang byte konsisten dengan offset karakter."""
    return data_bytes.decode("utf-8", errors="replace").encode("utf-8")


def train(data_bytes, vocab_size=DEFAULT_VOCAB, path=None):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=vocab_size,
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                             show_progress=False)
    text = canon_bytes(data_bytes).decode("utf-8")
    tok.train_from_iterator([text], trainer=tr)
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tok.save(path)
    return tok


def load(path):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(path)


def get(data_bytes, vocab_size=DEFAULT_VOCAB, cache=None):
    """Muat dari cache bila ada, selain itu latih (lalu simpan)."""
    if cache and os.path.exists(cache):
        return load(cache)
    return train(data_bytes, vocab_size, cache)


def encode(tok, data_bytes):
    """-> (ids int64, byte_len per token int64, bytes kanonik).

    Token disimpan dalam ruang byte->unicode (gaya GPT-2), jadi JUMLAH KARAKTER string token
    sama dengan jumlah byte aslinya. Jangan pakai offset karakter (bisa tumpang tindih) atau
    len(t.encode('utf-8')) (karakter non-ASCII jadi >1 byte).
    """
    cb = canon_bytes(data_bytes)
    text = cb.decode("utf-8")
    enc = tok.encode(text, add_special_tokens=False)
    ids = np.asarray(enc.ids, dtype=np.int64)
    lens = np.fromiter((len(t) for t in enc.tokens), np.int64, len(enc.tokens))
    return ids, lens, cb


def decode(tok, ids):
    return tok.decode([int(i) for i in ids], skip_special_tokens=False)
