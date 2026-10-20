"""Laporan HTML Tingkat 2 (Tingkat B — neural paralel-chunk).

Sumber: results/tierb_train.jsonl · tierb_train_tau.jsonl · tierb_integrate.jsonl ·
tierb_gen_250k.txt · tierb_gen_250k_tau.txt
Jalan: python scripts/make_tierb_report.py
"""
import html
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def jl(name):
    p = os.path.join(ROOT, "results", name)
    out = []
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def txt(name):
    p = os.path.join(ROOT, "results", name)
    return open(p, encoding="utf-8").read() if os.path.exists(p) else ""


def main():
    base = jl("tierb_train.jsonl")
    tau = jl("tierb_train_tau.jsonl")
    integ = jl("tierb_integrate.jsonl")

    parts = ["""<!DOCTYPE html><html lang="id"><head><meta charset="utf-8">
<title>Tingkat 2 (B) — neural paralel-chunk</title>
<style>
body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px;margin:2rem auto;
padding:0 1.2rem;color:#1a1a1a;line-height:1.6}
h1{border-bottom:3px solid #0a7;padding-bottom:.4rem}
h2{margin-top:2rem;color:#075}
table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.92rem}
th,td{border:1px solid #ccc;padding:.4rem .6rem;text-align:right}
th{background:#eef;text-align:center}
td:first-child,th:first-child{text-align:left}
pre{background:#f7f7f7;border:1px solid #ddd;padding:.8rem;border-radius:5px;
white-space:pre-wrap;font-size:.86rem;line-height:1.45}
code{background:#f4f4f4;padding:.1rem .35rem;border-radius:3px}
.box{background:#fff8e1;border-left:4px solid #f0a000;padding:.8rem 1rem;margin:1rem 0;border-radius:4px}
.good{background:#e8f5e9;border-left:4px solid #2e7d32;padding:.8rem 1rem;margin:1rem 0;border-radius:4px}
.bad{background:#ffebee;border-left:4px solid #d32f2f;padding:.8rem 1rem;margin:1rem 0;border-radius:4px}
</style></head><body>"""]

    parts.append("<h1>Tingkat 2 (B) — pelengkap neural paralel-chunk</h1>")
    parts.append("<p><b>Saron</b> (konv kausal) + <b>Gong</b> (rekurensi linear peluruhan "
                 "data-dependen) + gerbang + SwiGLU + RMSNorm. Acuan numpy float64. "
                 "Dilatih teacher-forced pada byte CE, BPTT penuh.</p>")

    parts.append('<div class="box"><b>Temuan 1 — inisialisasi peluruhan (diperiksa, diukur).</b> '
                 'Dokumen §4.2 menuntut "τ_h per head tersebar geometris (2…4096)". Kode lama '
                 'memakai <code>bl = 0</code> → λ_h = sigmoid(0) = <b>0,5 untuk SEMUA head</b> → '
                 'horizon awal τ ≈ 1 langkah. Sebaran τ diimplementasikan '
                 '(<code>bl_h = ln(τ_h−1)</code>) dan <b>diukur</b>: pada 250k byte hasilnya '
                 '<b>2,7300 vs 2,6654</b> (bl=0) → <b>sedikit lebih buruk</b>, dan '
                 '<b>tidak</b> menghilangkan keruntuhan generasi. Karena λ tetap bergantung data '
                 '(<code>h@wl</code>), model bisa belajar memisahkan skala sendiri; bias hanya '
                 'titik awal. Default dikembalikan ke <code>False</code> (pemenang terukur), flag '
                 'tetap ada untuk diuji ulang di skala besar. + tes regresi '
                 '(<b>6 tes gamelan lolos</b>).</div>')

    parts.append('<div class="bad"><b>Temuan 2 — generasi Tingkat B runtuh ke tarikan tetap.</b> '
                 'Apa pun prompt-nya, setelah ~40 byte keluaran menjadi teks yang SAMA '
                 '("…Dayak Kalimantara dan Dayak menuku…"). Pengaruh prompt hilang — tanda state '
                 'rekuren kecil tidak mempertahankan konteks pada skala latih ini. '
                 '<b>Bukan bug train/eval</b>: jalur latih dan jalur generasi terbukti menghitung '
                 'fungsi yang SAMA (selisih 1,7e−15). Ini keterbatasan model, bukan kebocoran.</div>')

    # tabel mandiri
    parts.append("<h2>1. Tingkat B mandiri (250k byte, Indonesia)</h2>")
    rows = []
    for r in base:
        rows.append(["sebelum perbaikan (bl=0)", f"{r['bpb']:.4f}", f"{r['bps']:.0f}",
                     f"{r['epochs']}"])
    for r in tau:
        rows.append(["<b>sesudah perbaikan (τ geometris)</b>", f"<b>{r['bpb']:.4f}</b>",
                     f"{r['bps']:.0f}", f"{r['epochs']}"])
    if rows:
        parts.append("<table><thead><tr><th>konfigurasi</th><th>BPB tahan</th>"
                     "<th>B/s</th><th>epoch</th></tr></thead><tbody>"
                     + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
                     + "</tbody></table>")
    parts.append("<p>Pembanding: <b>Tingkat A</b> pada korpus yang sama (200k/8k) = "
                 "<b>2,3261</b> bit. Tingkat B mandiri masih <b>kalah</b> dari Tingkat A.</p>")

    # integrasi A+B
    parts.append("<h2>2. Integrasi A+B dan gerbang G2 (≥1 jt byte)</h2>")
    if integ:
        r = integ[-1]
        keys = [k for k in r if k.startswith("A ")]
        base_v = r.get("A saja")
        rows = []
        for k in ["A saja"] + sorted(keys):
            if k in r:
                d = "" if k == "A saja" else f"{base_v - r[k]:+.4f}"
                rows.append([k, f"{r[k]:.4f}", d])
        parts.append("<table><thead><tr><th>varian</th><th>BPB</th>"
                     "<th>Δ vs A saja</th></tr></thead><tbody>"
                     + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in rr) + "</tr>"
                               for rr in rows)
                     + "</tbody></table>")
        d_tr = base_v - r.get("A + B terlatih", base_v)
        d_rnd = base_v - r.get("A + B acak", base_v)
        ok = d_tr >= 0.05
        cls = "good" if ok else "bad"
        parts.append(f'<div class="{cls}"><b>G2 {"LULUS" if ok else "BELUM LULUS"}</b> '
                     f'(ambang +0,05): kontribusi terlatih <b>{d_tr:+.4f}</b> bit. '
                     f'Kontrol B-acak {d_rnd:+.4f} '
                     f'({"⚠️ protokol bocor" if d_rnd > 0.005 else "≈0 → protokol bersih"}).</div>')
        parts.append('<p>Kontrol "kocok" (urutan masukan neural diacak) memastikan keuntungan '
                     'tidak datang dari kebocoran temporal.</p>')
    else:
        parts.append("<p><i>(belum ada hasil integrasi)</i></p>")

    # generasi
    parts.append("<h2>3. Generasi Tingkat B</h2>")
    g1 = txt("tierb_gen_250k.txt")
    g2 = txt("tierb_gen_250k_tau.txt")
    if g1:
        parts.append("<h3>Sebelum perbaikan (bl=0)</h3><pre>" + html.escape(g1) + "</pre>")
    if g2:
        parts.append("<h3>Sesudah perbaikan (τ geometris)</h3><pre>" + html.escape(g2) + "</pre>")

    parts.append("<h2>4. Kesimpulan</h2>")
    parts.append("""<ul>
<li>Tingkat B <b>dapat dilatih</b> dan paritas bentuk rekuren/chunkwise eksak (6 tes lolos).</li>
<li>Secara <b>kualitas mandiri ia kalah</b> dari Tingkat A pada korpus Indonesia (2,67 vs 2,33
pada 250k; dan 3,22 vs 3,22 pada 1M dalam integrasi).</li>
<li>Integrasi A+B pada <b>≥1 jt byte</b> (skala yang dokumen minta): Δ terlatih <b>≈ 0</b>,
kontrol acak ≈ 0 (protokol bersih), kontrol kocok negatif → neural membawa sinyal temporal tapi
<b>tidak menambah informasi</b> di atas tabel Tingkat A. <b>G2 belum lulus</b> (ambang +0,05).</li>
<li>Sesuai §7 dokumen: Tingkat B tetap <b>modul riset</b>, bukan jalur produksi; hasil negatif
ditulis (di sini) — inilah keluaran yang diminta gerbang itu.</li>
<li>Jalan yang belum dicoba untuk menaikkan B: d/H lebih besar, L&gt;1 blok bertumpuk,
pelatihan lebih lama (CE masih turun di akhir), Bonang (perhatian berjendela), dan backend
BLAS/GPU (sekarang ~1–3 kB/s, numpy float64 murni).</li>
</ul>""")

    parts.append("</body></html>")
    out = os.path.join(ROOT, "results", "report_tierb.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))
    print(f"ditulis -> {out}")


if __name__ == "__main__":
    main()
