# Flowchart lengkap TMT2

Dari input mentah sampai output dan pembaruan parameter. Semua nama variabel/fungsi diambil
langsung dari kode. Tanda **[AKTIF]** = dipakai di konfigurasi terbaik
(`head="linear"` + `deep_grad` + `only_byte_loss`); **[MATI]** = ada di kode tetapi tidak aktif.

Konfigurasi terbaik: `head="linear"`, `deep_grad=True`, `only_byte_loss=True`, `lr=1e-2`,
`warmup=200`, `mem_query_x=True`, `mem_wq_match_k=True`, `mem_hetero=True`,
`mem_surprise_fixed=True`, `beta_max=1.0`, `deep_grad_gate=True`, `think_prob=0.0`,
**`mixer_k=1` + `mixer_match_orders="1..12"` + `mixer_match=True`** (`Config.best()`).

> **Revisi 2026-10-04.** Jalur MIXER ditambahkan ke diagram — ia membawa **seluruh** kualitas
> (mixer-saja 1,8434 vs SSM-saja 2,9235 vs keduanya 1,8440 pada WikiText 200k). Lihat
> `ARCHITECTURE.md` §1b. Status tes: **77 lolos**.
>
> **Revisi 2026-10-06.** §1–§9 di bawah menggambarkan **TMT2 lama** (SSM + RTRL, `core.py`).
> **Jalur produksi sekarang adalah TMT2 v2 Tingkat A** (`tmt2/stat_core.py`) — alurnya
> **berbeda** dan digambarkan di **§10** di bawah. Status tes: **93 lolos**.

---

## 1. Alur satu byte — input → proses → output

```mermaid
flowchart TD
    START([byte b_t tiba dari stream]) --> PREV["ambil entri sebelumnya<br/>prev = s.buf[s.t]"]
    PREV --> CEPRE["ce_pre = CE(prev.logits, b_t)<br/>prediksi PREKUENSIAL<br/>(dihitung SEBELUM byte diproses)"]
    CEPRE --> ZSTAR["zstar = tdict[b_t]<br/>target encoder (EMA dari E)"]
    ZSTAR --> ENC["x = E[b_t]<br/>encoder: baris ke-b dari tabel E (V x d)"]

    ENC --> NG{"ngram aktif?<br/>ngram_lam &gt; 0 atau ngram_learn"}
    NG -->|ya| NGU["perbarui OnlineNgramN<br/>prior = log P_ngram"]
    NG -->|tidak| MIXQ{"mixer aktif?<br/>m.mix_w tidak None"}
    NGU --> MIXQ
    MIXQ -->|ya| MIXB["PERBARUI TABEL MIXER (SETELAH prediksi -> tanpa bocor)<br/>logpriors = [logprior_order(ctx, o) untuk tiap orde]<br/>+ logprior_match (deterministik, bila mixer_match)<br/>+ logprior_match_recent (indeks K-gram, bila diaktifkan)<br/>catat byte ke s.hist TEPAT SEKALI (flag appended)"]
    MIXQ -->|tidak| THINK
    MIXB --> THINK

    THINK{"think?<br/>rng &lt; think_prob"}
    THINK -->|ya| KMAX["K = think_max"]
    THINK -->|tidak| KZERO["K = 0"]
    KMAX --> CHAIN
    KZERO --> CHAIN

    CHAIN["_chain: tumpukan SSM<br/>+ logits += Σ_j w_j · logpriors_j<br/>(lihat bagian 2)"]
    CHAIN --> ENTRY["Entry e:<br/>chain, logits, stops, halt, k,<br/>zn = l2n(zstar), prior"]
    ENTRY --> PUB["e.byte = b_t<br/>r.byte = b_t untuk routing g_x ke E[b_t]"]
    PUB --> ADV["s.t += 1 ; s.buf[s.t] = e<br/>s.st = e.chain_state"]

    ADV --> TRAIN{"train?"}
    TRAIN -->|ya| RESOLVE["_resolve: hitung loss + backward<br/>(lihat bagian 3)"]
    TRAIN -->|tidak| SKIP["lewati belajar<br/>(eval/generate)"]
    RESOLVE --> AUX{"only_byte_loss?"}
    AUX -->|tidak| AUXL["_state_cap + _vic<br/>[MATI di konfig terbaik]"]
    AUX -->|ya| CONS
    AUXL --> CONS["_consume: perbarui parameter<br/>(lihat bagian 4)"]
    CONS --> GC["buang entri lama:<br/>hapus j &lt; t - maxH - 1"]
    SKIP --> GC
    GC --> OUT

    OUT["OUTPUT: ce_pre<br/>CE prekuensial untuk byte b_t"]
    OUT --> ONLINE{"s.online dan train?"}
    ONLINE -->|ya| SEG["s.seg.append(b_t) ; n_bytes += 1"]
    ONLINE -->|tidak| END([selesai])
    SEG --> SEGFULL{"len(s.seg) &gt;= seg_len?"}
    SEGFULL -->|ya| REPLAY["_replay_add(s)<br/>simpan segmen + snapshot_state<br/>(salinan MANDIRI h/M/kp)"]
    SEGFULL -->|tidak| SLEEPQ
    REPLAY --> SLEEPQ{"n_bytes % sleep_every == 0?"}
    SLEEPQ -->|ya| SLEEP["sleep(): latih ulang item replay<br/>(lihat bagian 5)"]
    SLEEPQ -->|tidak| END
    SLEEP --> END
```

---

## 2. `_chain` — tumpukan + loop berpikir

```mermaid
flowchart TD
    C0([masuk: x, zstar, K, prior, logpriors]) --> HEAD{"head linear?"}
    HEAD -->|ya| D0["D = None<br/>(kamus keluaran tak dipakai)"]
    HEAD -->|tidak| D1["D = l2n(dict_tensor())<br/>[MATI]"]
    D0 --> LOOP
    D1 --> LOOP

    LOOP["untuk k = 0 .. K"] --> STACK["_stack_step(xin, cur, z)<br/>(lihat bagian 6)"]
    STACK --> P1["p1 = top.y @ P[(Ltop,1)]"]
    P1 --> LG["logits[k] = W_out @ l2n(p1) + byte_bias"]
    LG --> PRIOR{"prior n-gram ada?"}
    PRIOR -->|ya| LAMP["logits[k] += lam * prior"]
    PRIOR -->|tidak| MIXA
    LAMP --> MIXA{"mix_w ada dan logpriors?"}
    MIXA -->|ya| MIXB["logits[k] += Σ_j w_j · logpriors_j<br/>PILAR 2 — mixer per-orde + match<br/>logp_j KONSTAN thd parameter -> gradien hanya ke w"]
    MIXA -->|tidak| STOPF
    MIXB --> STOPF["stops[k] = stop_fwd(top.y, ws, bs)<br/>[dipakai hanya bila use_stop]"]
    STOPF --> KCHK{"k == K?"}
    KCHK -->|ya| HALTL
    KCHK -->|tidak| HALTC{"evaluasi dan stop &gt; 0,5?"}
    HALTC -->|ya| HALTL
    HALTC -->|tidak| NEXT["xin = ln(p1) + flag<br/>z = None (inovasi mati di iterasi lanjut)"]
    NEXT --> LOOP

    HALTL["halt = halting(stops)<br/>PonderNet geometrik"] --> KSTAR["kstar =<br/>latih: sample dari halt<br/>eval: argmax"]
    KSTAR --> BUILD["Entry: chain, logits, stops, halt,<br/>k=kstar, chain_state = states[kstar]"]
    BUILD --> CEND([keluar])
```

---

## 3. `_resolve` — loss dan backward

```mermaid
flowchart TD
    R0([masuk: s, t]) --> GET["et = s.buf[t]<br/>ei = s.buf[t-1]"]
    GET --> JEPA{"only_byte_loss?"}
    JEPA -->|tidak| J1["JEPA lokal horizon 1/4/16:<br/>jepa_bwd(rec.y @ P, span_target)<br/>[MATI]"]
    JEPA -->|ya| BYTE
    J1 --> J2["JEPA kedalaman berpikir<br/>[MATI]"]
    J2 --> FLUSHJ["_flush(touched, 'jepa')<br/>[MATI]"]
    FLUSHJ --> BYTE

    BYTE["untuk tiap kedalaman k:<br/>glo = softmax_grad_target(logits[k], b, halt[k])"]
    BYTE --> GMIX["gradien MIXER (bila mix_w ada):<br/>dCE/dw_j = glo · logpriors_j<br/>g_mixw[j] += ... <br/>logp_j konstanta -> TIDAK ada gradien ke SSM dari sini"]
    GMIX --> HBL["head_bwd_lin_acc(p1, W_out, glo)<br/>-> g_y, g_P, g_byte_bias<br/>scatter langsung ke arena"]
    HBL --> GHL["top.gh = g_y<br/>(dL/dy lapisan puncak)"]
    GHL --> FLUSHB["_flush(top_recs, 'byte', deep=deep_grad)"]
    FLUSHB --> NOTE["_note('byte', dot(halt, ces))"]
    NOTE --> STOPT{"use_stop dan bukan only_byte?"}
    STOPT -->|ya| STOPL["stop_grads -> ws, bs<br/>[MATI]"]
    STOPT -->|tidak| REND([selesai])
    STOPL --> REND
```

### 3b. `_flush` — backward readout + kontraksi RTRL

```mermaid
flowchart TD
    F0([masuk: recs, task, deep]) --> DEEP{"deep?"}
    DEEP -->|ya| COLLECT["telusuri rec.below<br/>kumpulkan SEMUA lapisan<br/>urut puncak -> bawah"]
    DEEP -->|tidak| SORT
    COLLECT --> SORT["untuk tiap rec:"]

    SORT --> OX["ox = offset E[b] bila lapisan 0,<br/>selain itu -1"]
    OX --> FL["flush_rec(h, x, M, gy, ...)<br/>lihat 3c"]
    FL --> BELOW{"deep dan rec.l &gt; 0?"}
    BELOW -->|ya| ADD["b.gh += gy + o_gx<br/>gy = suku residual y = x + f<br/>o_gx = dL/dh . (dh/dx)  [BARU]<br/>[bila deep_grad_gate]"]
    BELOW -->|tidak| MW
    ADD --> MW{"mem_write_grad?"}
    MW -->|ya| MWB["mem_write_bwd: gradien Wk/Wv<br/>(terpangkas 1 langkah)<br/>[MATI, hasil negatif]"]
    MW -->|tidak| FEND([selesai])
    MWB --> FEND
```

### 3c. `flush_rec` — isi satu backward lapisan

```mermaid
flowchart TD
    FR0([masuk: h, x, M, gy, parameter, S, a, cc]) --> RECOMP["HITUNG ULANG forward readout:<br/>hn = LayerNorm(h)<br/>c0 = hn @ Wc<br/>uq = hn @ Wq ; q = l2n(uq)<br/>am = M @ q ; cf = c0 + am @ Wm<br/>z1 = cf @ Wg1 ; z2 = cf @ Wg2<br/>gv = z1*sigmoid(z1)*z2"]
    RECOMP --> BGLU["mundur GLU:<br/>g_Wo += outer(gv, gy)<br/>ga, gb, gc dari turunan z1,z2"]
    BGLU --> BMEM["mundur memori:<br/>tq = Wm^T gcf  (g_am)<br/>g_Wm += outer(am, gcf)<br/>g_u -> g_Wq += outer(hn, tq)<br/>ghn += Wq tq  [hanya bila BUKAN query_x]"]
    BMEM --> BWC["mundur Wc:<br/>g_Wc += outer(hn, gcf)<br/>ghn += Wc^T gcf"]
    BWC --> BLN["mundur LayerNorm:<br/>c0 = inv*(ghn - mean - hn*mean(ghn*hn))"]
    BLN --> GX["o_gx[j] = sum_i c0[i]*(cc_i*Wa[j,i] + (1-a_i)*B[j,i])<br/>d h_i / d x_j  [BARU]"]
    GX --> RTRL["KONTRAKSI RTRL:<br/>g_Wa += S_Wa * c0<br/>g_B += S_B * c0<br/>g_K += S_K * c0  (bila useK)<br/>g_ba += S_ba * c0"]
    RTRL --> GXF["g[ox+i] += gy[i] + o_gx[i]<br/>(y = x + f memberi gy)"]
    GXF --> FREND([selesai])
```

---

## 4. `_consume` — pembaruan parameter

```mermaid
flowchart TD
    CO0([masuk: s, fast_ok]) --> FAST{"fast_ok dan<br/>only_mask None?"}
    FAST -->|ya| FASTL["JALUR CEPAT (tiap byte):<br/>untuk tiap p di self.fast (Af, Bf):<br/>g = w(task) * ftg[p]<br/>nn = sqrt(g.g)<br/>p.data *= (1 - fast_decay)<br/>p.data -= fast_lr/nn * g"]
    FAST -->|tidak| CLEAR["kosongkan ftg"]
    FASTL --> NACC
    CLEAR --> NACC["nacc += 1"]
    NACC --> SLOWQ{"nacc &gt;= slow_every?"}
    SLOWQ -->|tidak| COEND([selesai])
    SLOWQ -->|ya| SLOW["slow_update()<br/>(lihat bagian 5)"]
    SLOW --> COEND
```

---

## 5. `slow_update` + `sleep`

```mermaid
flowchart TD
    SU0([masuk: slow_update]) --> W["wt = task berbobot (W_TASKS)"]
    W --> ONE{"len(wt) == 1?"}
    ONE -->|ya| FASTG["gb = gacc[task] / nacc * weight<br/>[AKTIF: hanya task byte]"]
    ONE -->|tidak| PCG["vektor per task<br/>PCGrad bila cfg.pcgrad<br/>[MATI: cuma 1 task]"]
    FASTG --> FTASK
    PCG --> FTASK["tambahkan task F_TASKS<br/>(jepa/vic/stop/state) [MATI]"]
    FTASK --> CLIP["gn = |gb| ; bila gn &gt; clip:<br/>gb *= clip/gn"]
    CLIP --> MASK{"only_mask ada?"}
    MASK -->|ya| MASKF["gb *= only_mask<br/>(isolasi only_params)"]
    MASK -->|tidak| ADAM
    MASKF --> ADAM

    ADAM["AdamW IN-PLACE:<br/>lr = cfg.lr * min(1, t_adam/warmup)<br/>m = b1*m + (1-b1)*gb<br/>v = b2*v + (1-b2)*gb^2<br/>upd = m_hat / (sqrt(v_hat) + eps)"]
    ADAM --> APPLY["untuk tiap nama di self.touched:<br/>theta[name] *= (1 - lr*wd)<br/>theta[name] -= lr * lr_mult[name] * upd<br/>(lr_mult: mix_w pakai mixer_lr_mult=0,1;<br/>parameter keluaran pakai out_lr_mult)"]
    APPLY --> UNC{"cfg.uncertainty?"}
    UNC -->|ya| UNCU["perbarui bobot uncertainty<br/>[MATI: cuma 1 task]"]
    UNC -->|tidak| EMA
    UNCU --> EMA["ema_update: tp[E] = ema*tp[E] + (1-ema)*E<br/>tdict = target_dict()"]
    EMA --> DECAY["decay_traces(s_decay) untuk tiap state<br/>S = {k: v * s_decay}"]
    DECAY --> SUEND([selesai])

    SL0([masuk: sleep]) --> ITEM["untuk tiap item di buffer replay:<br/>pulihkan state dari snapshot<br/>latih ulang segmen (gradien lambat)"]
    ITEM --> SLEEPEND([selesai])
```

---

## 6. `_stack_step` — satu langkah tumpukan (isi satu lapisan)

```mermaid
flowchart TD
    S0([masuk: x, st, z_star]) --> ZS["zs = l2n(z_star)"]
    ZS --> LOOPL["untuk l = 0 .. n_layers-1:"]
    LOOPL --> INNOV{"use_innovation dan<br/>zs dan prev_p1[l] ada?"}
    INNOV -->|ya| INNO["diff = zs - prev_p1[l]<br/>e = sqrt(d) * diff<br/>surprise = 0.5 * diff.diff"]
    INNOV -->|tidak| NOINN["e = 0, surprise = 0"]
    INNO --> SFWD
    NOINN --> SFWD["step_fwd: REKURENSI + RTRL + READOUT + TULIS MEMORI<br/>(lihat bagian 7)"]
    SFWD --> MKST["bentuk LayerState baru:<br/>h, M, S = {Wa, B, K, ba}"]
    MKST --> REC["FRec(l, depth, x, h, y, M, S, useK, a, cc)"]
    REC --> LINK["r.below = recs[l-1]  (gradien antar-lapisan)<br/>r.prev_w = rec l-1 (penulis M ini)<br/>r.beta = beta_max * surprise"]
    LINK --> PP["prev_p1[l] = l2n(y @ P[(l,1)])"]
    PP --> NEXT["xin = y  (STOP-GRAD antar lapisan)"]
    NEXT --> LOOPL
    LOOPL --> SEND([keluar: recs, StackState baru])
```

---

## 7. `step_fwd` — isi satu langkah lapisan

```mermaid
flowchart TD
    ST0([masuk: x, h_prev, S, parameter, M, kp_prev]) --> REC["REKURENSI:<br/>a_i = sigmoid(sum_j x_j Wa[j,i] + ba_i)<br/>u_i = sum_j x_j B[j,i]<br/>cc_i = (h_prev_i - u_i) * a_i * (1-a_i)"]
    REC --> HNEW["h_i = a_i*h_prev_i + (1-a_i)*u_i"]
    HNEW --> INOVE{"use_e?"}
    INOVE -->|ya| HINOV["h_i += sum_j e_j K[j,i]"]
    INOVE -->|tidak| TRACE
    HINOV --> TRACE["PERBARUI JEJAK RTRL:<br/>S_Wa[j,i] = a_i*S_Wa[j,i] + x_j*cc_i<br/>S_B[j,i]  = a_i*S_B[j,i]  + x_j*(1-a_i)<br/>S_K[j,i]  = a_i*S_K[j,i]  + e_j<br/>S_ba[i]   = a_i*S_ba[i]   + cc_i"]

    TRACE --> LN["READOUT - LayerNorm:<br/>hn = (h - mean(h)) / sqrt(var + eps)"]
    LN --> WC["c = hn @ Wc"]
    WC --> MEMR{"use_memory?"}
    MEMR -->|ya| QR["KUERI BACA:<br/>bila mem_query_x: q = l2n(ln(x) @ Wq)<br/>selain itu: q = l2n(hn @ Wq)"]
    MEMR -->|tidak| GLU
    QR --> AM["am = M @ q<br/>cf = c + am @ Wm"]
    AM --> GLU["GLU:<br/>z1 = cf @ Wg1 ; z2 = cf @ Wg2<br/>gv = z1 * sigmoid(z1) * z2<br/>f = gv @ Wo"]
    GLU --> ADAPT{"use_ad?"}
    ADAPT -->|ya| ADL["f += (cf @ Af) @ Bf"]
    ADAPT -->|tidak| YOUT
    ADL --> YOUT["y = x + f  (residual)"]

    YOUT --> MEMW{"use_memory?"}
    MEMW -->|tidak| SFEND
    MEMW -->|ya| KLN["ln(x) = LayerNorm(x) ; o_kp = ln(x)"]
    KLN --> KV["k = l2n(src @ Wk)  dengan<br/>src = kp_prev (hetero) atau ln(x)<br/>v = ln(x) @ Wv"]
    KV --> BETA["beta = beta_max * min(1, surprise)<br/>atau beta_max bila mem_surprise_fixed"]
    BETA --> DELTA["M <- M + beta * outer(v - M@k, k)<br/>(aturan delta-rule)"]
    DELTA --> SFEND([keluar: h, S, y, M, kp, a, cc])
```

---

## 8. Siklus hidup state

```mermaid
flowchart LR
    A["StackState<br/>per lapisan: h, M, S, kp, rec"] --> B["setiap byte:<br/>step_fwd menghasilkan state BARU<br/>h, M, S diperbarui"]
    B --> C["s.st = chain_state[kstar]<br/>(kedalaman terpilih)"]
    C --> D["setiap seg_len byte:<br/>snapshot_state -> seg_st0<br/>h/M/kp DISALIN (bukan referensi)<br/>jejak S DInolkan"]
    D --> E["_replay_add: segmen + snapshot<br/>masuk buffer replay"]
    E --> F["setiap sleep_every:<br/>sleep() memulihkan snapshot<br/>dan melatih ulang segmen"]
    F --> B
    C --> G["setiap slow_update:<br/>decay_traces(s_decay)<br/>S *= s_decay (pemotongan kredit)"]
    G --> C
```

**Peringatan penting:** `snapshot_state` WAJIB menyalin `h`, `M`, `kp`. `_BufPool` adalah ring
buffer — kalau state disimpan sebagai referensi, buffer kolam akan menimpanya sebelum replay
membacanya. Ini pernah jadi bug nyata (lihat `STATUS.md` P0b lama).

---

## 9. Ringkasan titik masuk kode

| tahap | fungsi | berkas |
|---|---|---|
| satu byte | `FastLearner.step` | `fast_learner.py` |
| mixer konteks (tabel + log-prior) | `OnlineNgramN.logprior_order` / `logprior_match` / `logprior_match_recent` | `fast_learner.py` |
| tumpukan + berpikir | `_chain` → `_stack_step` | `fast_learner.py` |
| isi satu lapisan (maju) | `fp.step_fwd` | `fastpath.py` |
| loss + backward | `_resolve` → `_flush` → `fp.flush_rec` | `fast_learner.py`, `fastpath.py` |
| pembaruan parameter | `_consume` → `slow_update` | `fast_learner.py` |
| replay + tidur | `sleep` | `fast_learner.py` |
| acuan kebenaran | `core.layer_step`, `core.stack_step` | `core.py` |
| acuan autograd | `autograd.py` | `autograd.py` |

`tmt2/autograd.py` adalah **definisi kebenaran**; jalur numba wajib diverifikasi terhadapnya
(**77 tes**, eksak ~1e-15).

---

# TMT2 v2 — alur JALUR PRODUKSI (Tingkat A + C)

> Ditambahkan 2026-10-06. Semua nama fungsi dari `tmt2/stat_core.py`. Status tes: **93 lolos**.

## 10. Alur satu byte — Tingkat A (inti statistik biner) + C (mixer/APM)

```
byte b_t (aliran)
   │
   ├─► _push(b)  ──► hist[pos & (HBUF-1)] = b ; pos++      (HBUF = 64 byte riwayat)
   │
   └─► stat_step(b, …, train)        ← SATU panggilan @njit per byte
         │
         ├─ [1] HASH KONTEKS (SEKALI per byte, di LUAR loop bit)
         │      untuk tiap konteks c (orde 1..6,8,12,16,24 + kata + jarang):
         │        hctx[c] = FNV-1a(byte pada pola offset coff[c])
         │        havail[c] = cukup riwayat?
         │
         ├─ [2] MATCH (M2)
         │      mkey = _mix64(FNV-1a(minlen byte terakhir))
         │      msl  = mkey & mmask          (indeks slot match)
         │      mck  = (mkey >> 40) & mck_mask   ← FIX 2026-10-05 (dulu mkey & 255 = bit indeks)
         │      bila mchk[msl] == mck → mpred = mbyte[msl]
         │
         └─ untuk bi = 0..7:                 ← 8 keputusan BINER per byte
               bit = (b >> (7-bi)) & 1
               │
               ├─ [3] LOOKUP TABEL per konteks
               │      key = _mix64(hctx[c] ^ c0*GOLDEN ^ (c+1)*CCTX)  ← FIX 2026-10-05 (tambah c)
               │      sl = key & mask ; ck = (key >> 32) & ck_mask
               │      d = tab[sl]                       (SATU int64: (ck<<32)|(n0<<16)|n1)
               │      bila (d>>32)==ck → st[c] = stretch((n1+0.5)/(n0+n1+1))
               │
               ├─ [4] MASUKAN match (st[nctx]) + neural (st[nctx+1]) + bias
               │
               ├─ [5] MIXER-2 (Tingkat C)
               │      lapis 1: 3 set bobot, dipilih c0 / prev_byte / (prev_byte,c0)
               │      lapis 2: campuran ketiganya, dipilih c0 → pm = sigmoid(dot)
               │
               ├─ [6] APM/SSE 2 tahap (DITERAPKAN saat eval, tabelnya saja yang dibekukan)
               │      pf = (pm + apm1[c0,i] + apm2[prev_byte,c0,i]) / 3
               │
               ├─ [7] CE += -log2(pf atau 1-pf)      ← per BIT; BPB = CE/8
               │
               └─ [8] bila train: perbarui mixer (galat AKHIR), APM, penghitung tab[sl]
                       count_cap → pangkas (a>>1, b>>1) saat jenuh
               c0 = c0*2 + bit            (prefiks biner untuk bit berikutnya)

BPB = rata-rata CE per byte = CE_bits / 8
```

**Titik masuk kode (v2)**

| tahap | fungsi | berkas |
|---|---|---|
| satu byte | `StatCore.step` | `stat_core.py` |
| kernel inti (semua langkah di atas) | `stat_step` | `stat_core.py` |
| konteks (pola offset + kata) | `build_contexts` | `stat_core.py` |
| evaluasi beku (tabel beku, konteks jalan) | `StatCore.eval_stream` | `stat_core.py` |
| generasi (sampel dari pohon biner) | `StatCore.generate` → `_sample_byte` | `stat_core.py` |
| latih 30 jt + generasi | `main` | `scripts/train_gen_big.py` |
| investigasi kurva-U (per blok vs bz2) | `main` | `scripts/diag_blocks.py` |

## 11. Alur Tingkat B (neural paralel-chunk) — modul riset (G2 gagal)

```
byte b_t ──► E[b] (embedding 256×d)
              │
              ├─► RMSNorm
              ├─► SARON  : konv kausal depthwise (k=8) + SiLU + pointwise
              │            state (k-1, d) dibawa antar-chunk   [paritas eksak]
              ├─► GONG   : U_t = λ_t ⊙ U_{t-1} + (1-λ_t) k_t v_tᵀ ; o_t = q_t U_t
              │            λ_t = sigmoid(h·Wl + bl)   ← bl: sebaran τ opsional (tau_spread)
              │            mode "chunk" (latih, matmul) == mode "recurrent" (generasi)
              ├─► gerbang per head (Saron vs Gong)
              ├─► SwiGLU + residual + RMSNorm
              └─► logits = W_out · y   →  p(bit|prefiks) via pohon biner
                                           → satu masukan mixer Tingkat A

Latih: BPTT penuh (rekuren; gradien identik dgn chunkwise). ~1–3 kB/s (numpy float64).
Status: G2 GAGAL @1 jt byte (Δ −0,0005). Generasi runtuh ke tarikan tetap. Modul riset.
```

| tahap | fungsi | berkas |
|---|---|---|
| satu blok (paralel / sekuensial) | `GamelanBlock.forward` | `gamelan/block.py` |
| satu langkah decode | `GamelanBlock.step` | `gamelan/block.py` |
| rekurensi Gong (2 bentuk) | `gong_recurrent` / `gong_chunkwise` | `gamelan/gong.py` |
| konv kausal Saron | `saron_forward` | `gamelan/saron.py` |
| maju + mundur (latih) | `block_forward_cache` / `block_backward` | `gamelan/train.py` |
| integrasi A+B + kontrol | `main` | `scripts/tierb_integrate.py` |
