"""Bangun laporan HTML investigasi korpus Indonesia besar.

Sumber: results/diag_overwrite.jsonl · diag_fix.jsonl · diag_component.jsonl ·
gen_big_id.jsonl · fair_compare_big.jsonl · extract_parquet.json
Jalan: python scripts/make_big_report.py
"""
import html
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def jl(name):
    p = os.path.join(ROOT, "results", name)
    if not os.path.exists(p):
        return []
    out = []
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def tbl(headers, rows):
    h = "".join(f"<th>{html.escape(str(x))}</th>" for x in headers)
    b = ""
    for r in rows:
        b += "<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
    return f'<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


def main():
    overwrite = jl("diag_overwrite.jsonl")
    fix = jl("diag_fix.jsonl")
    comp = jl("diag_component.jsonl")
    gen = jl("gen_big_id.jsonl")
    fair = jl("fair_compare_big.jsonl")

    parts = []
    parts.append("""<!DOCTYPE html><html lang="id"><head><meta charset="utf-8">
<title>Investigasi Korpus Indonesia Besar — Tingkat A</title>
<style>
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px;margin:2rem auto;
padding:0 1.2rem;color:#1a1a1a;line-height:1.6}
h1{border-bottom:3px solid #0a7;padding-bottom:.4rem}
h2{margin-top:2rem;color:#075}
table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.92rem}
th,td{border:1px solid #ccc;padding:.4rem .6rem;text-align:right}
th{background:#eef;text-align:center}
td:first-child,th:first-child{text-align:left}
code{background:#f4f4f4;padding:.1rem .35rem;border-radius:3px;font-size:.9em}
.box{background:#fff8e1;border-left:4px solid #f0a000;padding:.8rem 1rem;margin:1rem 0;border-radius:4px}
.bad{background:#ffebee;border-left:4px solid #d32f2f;padding:.8rem 1rem;margin:1rem 0;border-radius:4px}
.good{background:#e8f5e9;border-left:4px solid #2e7d32;padding:.8rem 1rem;margin:1rem 0;border-radius:4px}
.best{font-weight:700;color:#075}
</style></head><body>""")

    parts.append("<h1>Investigasi Korpus Indonesia Besar (190 MB) — Tingkat A</h1>")
    parts.append("<p>Korpus: <code>data/indonesian_big.bin</code> — 190.330.359 byte, "
                 "670.424 dokumen dari <code>IndonesiNewData.parquet</code> "
                 "(filter <code>language=='id'</code>, panjang 80–20.000 karakter).</p>")

    parts.append('<div class="bad"><b>⚠️ Kebocoran protokol keempat (ditemukan &amp; diperbaiki).</b> '
                 'Versi pertama <code>diag_overwrite.py</code> memakai jendela uji = '
                 '<b>ekor prefix latih</b> → model dinilai pada data yang sudah dilihat. '
                 'Terukur BPB palsu <b>0,5281</b> padahal jujur <b>2,1361</b> (rasio 4×). '
                 'Kontrol pembuktian: jendela-latih 0,5279 vs jendela-tak-terlihat 2,1361. '
                 'Diperbaiki + assert + 2 tes penjaga. <b>92 tes lolos.</b></div>')

    parts.append('<div class="good"><b>✅ Dua cacat kode nyata ditemukan &amp; diperbaiki.</b> '
                 '<b>(1) Checksum match = bit indeks</b> — <code>mck = mkey &amp; 255</code> '
                 'selalu sama dengan slot yang dicocokkan (terbukti pada <b>100% dari 299.994 '
                 'sampel</b>), jadi verifikasi match tak pernah menolak. '
                 '<b>(2) Indeks konteks <code>c</code> tidak ikut di-hash</b> — dua konteks '
                 'berbeda dengan pola offset sama mendapat kunci sama. '
                 'Diperbaiki: <code>mck = (mkey &gt;&gt; 40) &amp; mck_mask</code> dan '
                 '<code>key = _mix64(hctx[c] ^ (c0*GOLDEN) ^ (c+1)*CCTX)</code>. '
                 'Efek gabungan terukur (tb=22, 5 jt byte latih): <b>2,6371 → 2,5755 = −0,0616 '
                 'bit</b> (match_fix −0,047 · ctx_fix −0,015). <b>92 tes lolos.</b></div>')

    # --- kurva U
    parts.append("<h2>1. Kurva-U: BPB-uji memburuk seiring latihan</h2>")
    parts.append("<p>Jendela uji 20.000 byte <b>setelah</b> prefix latih (tak beririsan). "
                 "BPB diukur pada tiap 3 juta byte latih, untuk tiga ukuran tabel.</p>")
    bytb = {}
    for r in overwrite:
        bytb.setdefault(r["table_bits"], {})[r["n"]] = r
    ns = sorted({r["n"] for r in overwrite})
    rows = []
    for n in ns:
        cells = [f"{n:,}"]
        for tb in sorted(bytb):
            r = bytb[tb].get(n)
            v = f"{r['bpb']:.4f}" if r else "—"
            cells.append(v)
        rows.append(cells)
    parts.append(tbl(["byte latih"] + [f"tb={tb}" for tb in sorted(bytb)], rows))
    parts.append('<div class="box"><b>Temuan:</b> semua ukuran tabel <b>konvergen ke ~2,28</b> '
                 'pada 30 juta byte — bahkan <code>tb=27</code> (1 GB). Tabel lebih besar hanya '
                 '<b>menunda</b> titik balik (tb=25 → 9 jt, tb=27 → 12 jt), bukan menghilangkannya. '
                 '<b>Pada korpus bergerak ini</b> ukuran tabel bukan penyebab tunggal. '
                 'Catatan penting: kurva-U <b>tidak tereproduksi</b> pada kontrol 1 MB yang '
                 'stasioner (tb=16, 15 byte/slot: CE-seg 3,09 → 2,90 monoton turun). Jadi '
                 '“bukan efek tekanan tabel” hanya berlaku kuat pada rezim stasioner — pada '
                 'korpus nyata, tekanan tabel tetap kandidat.</div>')

    # --- knob yang dibuang
    parts.append("<h2>2. Hipotesis yang DIBUANG secara empiris</h2>")
    if fix:
        bycap = {}
        for r in fix:
            bycap.setdefault(r["count_cap"], {})[r["n"]] = r["bpb"]
        ns2 = sorted({r["n"] for r in fix})
        rows = [[f"{cap}"] + [f"{bycap[cap].get(n, float('nan')):.4f}" for n in ns2]
                for cap in sorted(bycap)]
        parts.append("<p><b>(a) <code>count_cap</code></b> (ambang pemangkasan penghitung):</p>")
        parts.append(tbl(["count_cap"] + [f"{n:,} B" for n in ns2], rows))
        parts.append('<div class="box">1023 vs 4095 hampir identik (2,4832 vs 2,4832). '
                     '<b>Bukan penyebabnya.</b> ⚠️ Catatan metodologis: <code>COUNT_CAP</code> '
                     'sebagai konstanta tingkat-modul <b>DIABAIKAN</b> oleh numba (dibekukan saat '
                     'kompilasi) — angka identik itu awalnya artefak, bukan temuan. Setelah '
                     'dijadikan parameter fungsi, efeknya tetap ≈0.</div>')
    parts.append('<p><b>(b) <code>cks_bits</code></b> (lebar checksum) pada 200k byte: '
                 '8 bit <b>2,470495</b> · 16 bit <b>2,470172</b> · 24 bit <b>2,470173</b> '
                 '→ efek −0,0003 bit. <b>Bukan penyebabnya.</b></p>')
    parts.append('<p><b>(c) ukuran tabel</b> (<code>table_bits</code>) — lihat bagian 1.</p>')
    parts.append('<p><b>(d) kesulitan data naik.</b> bz2 -9 per blok 1 MB = '
                 '2,5511–2,5784 sepanjang 30 MB → <b>kesulitan data STABIL</b>. '
                 '<b>Dibuang.</b></p>')
    parts.append('<p><b>(e) drift/urutan dokumen.</b> Eksperimen 2 (<code>diag_normalized.py</code>): '
                 'latih pada urutan ASLI vs ACAK (dokumen diacak, 1% dokumen disisihkan sebagai '
                 'uji tersebar ≥190 kB). Hasil: <b>ASLI 2,1605→2,1434→2,1577→2,1720</b> · '
                 '<b>ACAK 2,1532→2,1650→2,1460→2,1714</b>. Kurva-U muncul di <b>KEDUANYA</b> '
                 'dengan bentuk nyaris identik (selisih akhir 0,0006 bit). → <b>urutan/drift '
                 'BUKAN penyebab.</b></p>')
    parts.append('<p><b>(f) <code>apm_lr</code></b>: −0,007 bit (kecil). '
                 '<b>(g) korelasi slot↔checksum</b>: ≈0. <b>Keduanya dibuang.</b></p>')

    # --- komponen
    parts.append("<h2>3. Ablasi komponen: siapa yang menanggung degradasi?</h2>")
    if comp:
        vs = ["penuh", "tanpa_apm", "mixer_seragam", "tabel_kosong"]
        rows = []
        for r in comp:
            cells = [f"{r['n']:,}"]
            for v in vs:
                if v in r:
                    cells.append(f"{r[v]:.4f}")
            rows.append(cells)
        parts.append(tbl(["byte latih"] + vs, rows))
        # hitung selisih APM
        delta = [(r["n"], r["tanpa_apm"] - r["penuh"]) for r in comp if "tanpa_apm" in r]
        s = " · ".join(f"{n//10**6}jt +{d:.4f}" for n, d in delta)
        parts.append(f'<div class="good"><b>Temuan kunci:</b> kolom <code>tanpa_apm</code> '
                     f'dikurangi <code>penuh</code> selalu <b>POSITIF</b>: {s}. '
                     'Nilai positif = BPB naik saat APM dimatikan = <b>APM MENOLONG</b>. '
                     'Manfaat APM bahkan <b>TUMBUH</b> seiring latihan (+0,0275 @5 jt → '
                     '+0,058 @30 jt). <b>APM/SSE bukan penyebab degradasi — jangan dimatikan.</b> '
                     '(Catatan: pembacaan awal yang menyebut “APM merugikan” <b>SALAH</b> dan '
                     'telah dikoreksi di sini.)</div>')
        parts.append('<p>Kolom <code>mixer_seragam</code> (mixer di-reset ke bobot seragam) '
                     '~2,81–2,90 vs <code>penuh</code> ~2,15–2,23: mixer <b>sangat berperan</b> '
                     '(menyumbang ~0,65 bit).</p>')

    # --- baseline
    if fair:
        parts.append("<h2>4. Baseline adil pada jendela uji yang sama</h2>")
        rows = []
        for r in fair:
            rows.append([html.escape(r["method"]),
                         f"{r.get('mode_beku', float('nan')):.4f}",
                         f"{r.get('mode_lihat', float('nan')):.4f}",
                         "ya" if r.get("online") else "tidak"])
        parts.append(tbl(["metode", "beku", "lihat", "online"], rows))
        parts.append('<p><i>beku</i> = hanya melihat data latih lalu menekan jendela uji '
                     '(out-of-sample). <i>lihat</i> = menekan jendela uji itu sendiri '
                     '(batas atas tidak adil).</p>')

    # --- generasi
    if gen:
        parts.append("<h2>5. Generasi</h2>")
        rows = [[f"{r['ntrain']:,}", f"{r['bpb']:.4f}", f"{r['bps']:.0f}",
                 f"{r['train_s']/60:.1f}", f"{r.get('table_fill', float('nan')):.3f}"]
                for r in gen]
        parts.append(tbl(["byte latih", "BPB uji", "B/s", "menit", "isi tabel"], rows))

    parts.append("<h2>6. Kesimpulan &amp; langkah berikutnya</h2>")
    parts.append("""<ul>
<li>Model Tingkat A <b>belajar dengan benar</b> pada korpus 190 MB: CE-seg dan BPB-uji turun
monoton di awal (2,0749 → 2,0663 pada 10→20 jt byte).</li>
<li>Namun pada skala puluhan juta byte muncul <b>kurva-U</b>: setelah tabel jenuh, BPB-uji
<b>naik</b> lagi (2,10 → 2,27).</li>
<li>Hipotesis yang <b>DIBUANG</b>: ukuran tabel (sebagai penyebab tunggal), <code>count_cap</code>,
lebar checksum, <code>apm_lr</code>, korelasi slot↔checksum, kesulitan data, dan
<b>urutan/drift dokumen</b> (eksperimen 2).</li>
<li><b>APM/SSE MENOLONG</b> (+0,0275 @5 jt → +0,058 @30 jt) — <b>JANGAN dimatikan</b>.
Pembacaan awal “APM merugikan” salah dan sudah dikoreksi.</li>
<li>Dua <b>cacat kode nyata</b> ditemukan &amp; diperbaiki (checksum match = bit indeks;
indeks konteks tak masuk hash), gabungan −0,0616 bit.</li>
<li>Tersangka yang <b>tersisa</b>: desain tabel direct-mapped (tabrakan/penimpaan), mesin
match, dan kalibrasi penghitung pada data bergerak (non-stasioner).</li>
<li>Untuk latihan 190 MB penuh, rekomendasi: <b>biarkan APM hidup</b>; pakai tabel lebih besar
(tb=25–27); dan jalankan eksperimen 1 (CE-seg ternormalisasi per blok, termasuk blok 30–190 MB)
+ eksperimen 3 (cangkok komponen di checkpoint 30 jt) sebelum menyimpulkan.</li>
</ul>""")

    parts.append("</body></html>")

    out = os.path.join(ROOT, "results", "big_corpus_report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"ditulis -> {out}")


if __name__ == "__main__":
    main()
