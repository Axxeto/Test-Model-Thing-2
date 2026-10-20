"""Ekstrak korpus parquet IndonesiaNewData -> aliran byte mentah ber-cache.

Kenapa perantara .bin?
  * Membaca parquet butuh pyarrow (berat, ~70 s untuk 838k baris) dan memuat SEMUA teks ke
    RAM sekaligus. Latih Tingkat A online hanya butuh SATU aliran byte, jadi cukup dibaca
    sekali lalu disimpan sebagai .bin; pelatihan berulang jadi instan.
  * Skema: text(string), doc_id(long), language(string), quality_score(float), Spark 4.1.1.

Filter (default):
  * language == 'id' (abaikan baris non-Indonesia yang jumlahnya kecil)
  * min_len <= len(text) <= max_len  (doc terlalu pendek = sampah, terlalu panjang = outlier)
  * quality_score >= min_quality
  * buang dokumen yang setelah strip kosong
Dokumen digabung dengan pemisah '\\n\\n' (pembatas alami ala paragraf).

PENTING — PROTOKOL UJI: jendela uji = EKOR aliran (byte terakhir). Skrip latih harus memakai
jendela yang sama persis. Karena kita menulis SATU berkas deterministik, hal itu otomatis.

Jalan:
  python scripts/extract_parquet.py                     # default: data/IndonesiNewData.parquet
  python scripts/extract_parquet.py --min_quality 0.5 --max_docs 200000
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np

from tmt2.corpus import sanity_check

SEP = "\n\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/IndonesiNewData.parquet")
    ap.add_argument("--out", default="data/indonesian_big.bin")
    ap.add_argument("--lang", default="id", help="kode bahasa yang dipakai ('id'; '' = semua)")
    ap.add_argument("--min_len", type=int, default=80, help="panjang dok minimum (karakter)")
    ap.add_argument("--max_len", type=int, default=20000, help="panjang dok maksimum (karakter)")
    ap.add_argument("--min_quality", type=float, default=0.0, help="ambang quality_score")
    ap.add_argument("--max_docs", type=int, default=0, help="0 = semua dokumen lolos filter")
    ap.add_argument("--max_bytes", type=int, default=0,
                    help="0 = semua; bila >0 berhenti setelah aliran mencapai n byte")
    ap.add_argument("--stats_out", default="results/extract_parquet.json")
    a = ap.parse_args()

    import pyarrow.parquet as pq

    pf = pq.ParquetFile(a.parquet)
    nrows = pf.metadata.num_rows
    print(f"[extract] {a.parquet} · {nrows} baris · {pf.metadata.num_row_groups} row group")

    keep = 0
    skipped_lang = skipped_len = skipped_q = skipped_empty = 0
    out = bytearray()
    t0 = time.perf_counter()

    # proses per-batch agar RAM rendah
    for batch in pf.iter_batches(batch_size=20000,
                                 columns=["text", "language", "quality_score"]):
        texts = batch.column("text").to_pylist()
        langs = batch.column("language").to_pylist()
        quals = batch.column("quality_score").to_pylist()
        for t, lg, q in zip(texts, langs, quals):
            if a.lang and lg != a.lang:
                skipped_lang += 1
                continue
            if q is None:
                q = 0.0
            if q < a.min_quality:
                skipped_q += 1
                continue
            if t is None:
                skipped_empty += 1
                continue
            t = t.strip()
            if len(t) < a.min_len or len(t) > a.max_len:
                skipped_len += 1
                continue
            if not t:
                skipped_empty += 1
                continue
            out += t.encode("utf-8") + SEP.encode("utf-8")
            keep += 1
            if a.max_docs and keep >= a.max_docs:
                break
            if a.max_bytes and len(out) >= a.max_bytes:
                break
        if (a.max_docs and keep >= a.max_docs) or (a.max_bytes and len(out) >= a.max_bytes):
            break
    dt = time.perf_counter() - t0

    data = bytes(out)
    print(f"[extract] simpan {keep} dok · {len(data)} byte · {dt:.1f} s")
    print(f"[extract] dibuang: bahasa={skipped_lang} panjang={skipped_len} "
          f"kualitas={skipped_q} kosong={skipped_empty}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "wb") as f:
        f.write(data)
    print(f"[extract] ditulis -> {a.out}")

    info = sanity_check(data, vocab=256, name="indonesian_big")
    # cuplikan awal untuk mata manusia
    head = data[:600].decode("utf-8", errors="replace")
    print("[extract] cuplikan awal:\n" + head.replace("\n", " ")[:400])

    os.makedirs(os.path.dirname(a.stats_out) or ".", exist_ok=True)
    rec = dict(parquet=a.parquet, out=a.out, nrows=nrows, docs_kept=keep,
               bytes_out=len(data), lang=a.lang, min_len=a.min_len, max_len=a.max_len,
               min_quality=a.min_quality, skipped=dict(lang=skipped_lang, length=skipped_len,
                                                       quality=skipped_q, empty=skipped_empty),
               h0=info["H0"], mode_frac=info["mode_frac"], seconds=dt)
    with open(a.stats_out, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"[extract] statistik -> {a.stats_out}")


if __name__ == "__main__":
    main()
