"""Bangun laporan HTML generasi Indonesia (sampel teks + metrik objektif).

  python scripts/make_gen_report.py
"""
import json, os, sys, html

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_txt(name, default=""):
    p = os.path.join(ROOT, name)
    return open(p, encoding="utf-8").read() if os.path.exists(p) else default


def parse_gen(txt):
    """-> list of (label, [(T, text)])"""
    blocks, cur, label = [], [], None
    for line in txt.splitlines():
        if line.startswith("-- ") and line.endswith(" --"):
            if label: blocks.append((label, cur))
            label, cur = line[3:-3].strip(), []
        elif ":" in line and line.strip().startswith("T="):
            try:
                T = line.split("T=")[1].split()[0]
                t = line.split(": ", 1)[1]
                cur.append((T, t))
            except Exception:
                pass
    if label: blocks.append((label, cur))
    return blocks


def esc(s):
    return html.escape(s)


def main():
    mix_txt = read_txt("results/gen_id_mixer.txt")
    ssm_txt = read_txt("results/gen_id_ssm.txt")
    mix_blocks = parse_gen(mix_txt)
    ssm_blocks = parse_gen(ssm_txt)
    q = {}
    p = os.path.join(ROOT, "results/gen_quality_3seed.jsonl")
    if os.path.exists(p):
        rows = [json.loads(l) for l in open(p)]
        for tag in ("mixer", "SSM"):
            grp = [r for r in rows if (tag in r["variant"]) == (tag == "mixer")]
            if grp:
                q[tag] = dict(
                    bpb=sum(r["test_bpb"] for r in grp) / len(grp),
                    gen=sum(r["gen_judge_bit"] for r in grp) / len(grp))
    real = rows[0]["real_judge_bit"] if rows else 1.676
    rand = rows[0]["rand_judge_bit"] if rows else 14.187

    def sample_rows(blocks, cls):
        out = []
        for label, items in blocks:
            for T, text in items:
                out.append(f'<tr><td class="lbl">{esc(label)}</td><td class="tmp">{esc(T)}</td>'
                           f'<td class="sm {cls}">{esc(text)}</td></tr>')
        return "".join(out)

    parts = []
    parts.append('<h2 class="sr-only">TMT2 generation test on Indonesian text with objective quality metrics</h2>')
    parts.append('<h3>TMT2 generasi bahasa Indonesia</h3>')
    parts.append('<p class="sub">Latih online 300.000 byte dari <code>data/indonesian_1m_pure.txt</code>. '
                 'Model byte-level, tanpa tokenizer. Metrik objektif: BPB uji-tahan + penilai n-gram-3 luar.</p>')

    parts.append('<div class="grid">')
    parts.append(f'<div class="metric"><div class="k">BPB uji-tahan (mixer)</div><div class="v">{q.get("mixer",{}).get("bpb",0):.3f}</div></div>')
    parts.append(f'<div class="metric"><div class="k">BPB uji-tahan (SSM-saja)</div><div class="v">{q.get("SSM",{}).get("bpb",0):.3f}</div></div>')
    parts.append(f'<div class="metric"><div class="k">Skor generasi (mixer)</div><div class="v">{q.get("mixer",{}).get("gen",0):.2f}</div></div>')
    parts.append(f'<div class="metric"><div class="k">Skor generasi (SSM-saja)</div><div class="v">{q.get("SSM",{}).get("gen",0):.2f}</div></div>')
    parts.append('</div>')

    parts.append(f'<p class="hint">Skor generasi = berapa bit/byte penilai n-gram-3 (dilatih pada teks uji-tahan) '
                 f'menilai keluaran model. <b>Rendah = bagus.</b> Anchor: teks Indonesia ASLI = {real:.2f} bit, '
                 f'byte ACAK = {rand:.2f} bit.</p>')

    parts.append('<div class="bar-wrap"><div class="bar-row"><span class="bar-lbl">Teks Indonesia asli</span>'
                 f'<div class="bar" style="width:{real/rand*100:.1f}%;background:#639922"></div><span class="bar-num">{real:.2f}</span></div>')
    gm = q.get("mixer", {}).get("gen", 2.2)
    gs = q.get("SSM", {}).get("gen", 3.2)
    parts.append(f'<div class="bar-row"><span class="bar-lbl">Generasi mixer</span>'
                 f'<div class="bar" style="width:{gm/rand*100:.1f}%;background:#378ADD"></div><span class="bar-num">{gm:.2f}</span></div>')
    parts.append(f'<div class="bar-row"><span class="bar-lbl">Generasi SSM-saja</span>'
                 f'<div class="bar" style="width:{gs/rand*100:.1f}%;background:#BA7517"></div><span class="bar-num">{gs:.2f}</span></div>')
    parts.append(f'<div class="bar-row"><span class="bar-lbl">Byte acak</span>'
                 f'<div class="bar" style="width:100%;background:#B4B2A9"></div><span class="bar-num">{rand:.2f}</span></div></div>')

    parts.append('<h4>Keluaran mixer (Config.best, orde 1..8)</h4>')
    parts.append('<table><thead><tr><th>prompt</th><th>T</th><th>keluaran</th></tr></thead><tbody>')
    parts.append(sample_rows(mix_blocks, "good"))
    parts.append('</tbody></table>')

    parts.append('<h4>Keluaran SSM-saja (mixer mati) — pembanding</h4>')
    parts.append('<table><thead><tr><th>prompt</th><th>T</th><th>keluaran</th></tr></thead><tbody>')
    parts.append(sample_rows(ssm_blocks, "bad"))
    parts.append('</tbody></table>')

    parts.append('<p class="hint">SSM-saja menangkap BENTUK imbuhan Indonesia (ber-, meng-, me-, -an) tetapi '
                 'tidak menghasilkan kata nyata — pseudo-morfologi. Mixer yang membuat generasi benar-benar bekerja.</p>')

    css = """<style>
.wrap{font-family:var(--font-sans);max-width:100%;color:var(--color-text-primary)}
h3{font-size:15px;font-weight:500;margin:0 0 4px}
h4{font-size:13px;font-weight:500;margin:1.5rem 0 8px}
.sub{font-size:13px;color:var(--color-text-secondary);margin:0 0 1rem;line-height:1.6}
.hint{font-size:12px;color:var(--color-text-secondary);line-height:1.6;margin:8px 0 0}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-bottom:1rem}
.metric{background:var(--color-background-secondary);border-radius:var(--border-radius-md);padding:1rem}
.metric .k{font-size:12px;color:var(--color-text-secondary);margin-bottom:4px}
.metric .v{font-size:24px;font-weight:500}
table{width:100%;border-collapse:collapse;table-layout:fixed}
th{font-size:12px;font-weight:500;text-align:left;color:var(--color-text-secondary);padding:6px 8px;border-bottom:0.5px solid var(--color-border-tertiary)}
td{font-size:12px;padding:7px 8px;border-bottom:0.5px solid var(--color-border-tertiary);vertical-align:top;line-height:1.5}
td.lbl{width:90px;color:var(--color-text-secondary)}
td.tmp{width:44px;color:var(--color-text-secondary)}
td.sm{font-family:var(--font-mono);word-break:break-word}
td.good{color:#185FA5}
td.bad{color:#854F0B}
.bar-wrap{margin:1rem 0 1.5rem}
.bar-row{display:flex;align-items:center;gap:10px;margin-bottom:6px}
.bar-lbl{width:150px;font-size:12px;color:var(--color-text-secondary);flex-shrink:0}
.bar-num{font-size:12px;font-weight:500;width:44px}
.bar{height:16px;border-radius:4px;flex:1;max-width:340px}
</style>"""
    html_out = "<div class=\"wrap\">" + css + "".join(parts) + "</div>"
    out = os.path.join(ROOT, "results", "gen_id_report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html_out)
    print("->", out)


if __name__ == "__main__":
    main()
