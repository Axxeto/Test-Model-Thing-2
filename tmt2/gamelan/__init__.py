"""TINGKAT B — pelengkap neural paralel-chunk (rancangan TMT2 v2, ARCHITECTURE_V2.md §4.2).

Paket ini berisi **acuan numpy float64**. Sesuai tradisi proyek: acuan dulu, jalur cepat
kemudian, dan setiap jalur cepat wajib lulus tes paritas terhadap acuan.

  `gong.py`  — rekurensi linear dengan peluruhan bergantung-data, DUA bentuk:
               rekuren (inferensi, O(1) per langkah) dan chunkwise (latihan, matmul per chunk).
  `saron.py` — konvolusi kausal + SiLU, dengan state yang terbawa antar-chunk.
  `block.py` — Saron + Gong + gerbang + SwiGLU + RMSNorm; evaluasi paralel = decode sekuensial.

Yang BELUM ada: Bonang (SWA+ALiBi, opsional), pelatihan BPTT per-chunk, kepala pohon biner
ke Tingkat C, dan backend cepat (BLAS batch / GPU). Lihat §8 dokumen rancangan.
"""
from .gong import gong_recurrent, gong_chunkwise
from .saron import saron_forward, saron_state_size
from .block import GamelanBlock

__all__ = ["gong_recurrent", "gong_chunkwise", "saron_forward", "saron_state_size",
           "GamelanBlock"]
