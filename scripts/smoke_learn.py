import sys, time, glob
sys.path.insert(0, ".")
import numpy as np
from tmt2.autograd import CTX, cross_entropy
from tmt2.model import Config, TMT2
from tmt2.core import StackState, stack_step, byte_logits, backward_task
from tmt2.corpus import license_bytes

# teks nyata (lintas-platform: data/licenses/ atau /usr/share/common-licenses/)
src = license_bytes()
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
text = src[:N]
cfg = Config()
if len(sys.argv) > 2: cfg = Config(d=int(sys.argv[2]))
lr_scale = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
model = TMT2(cfg)
print("params", model.n_params(), "bytes", len(text), "d", cfg.d)
tgt = model.target_dict()
st, prev = StackState(model), None
ces, t0 = [], time.time()
L = cfg.n_layers - 1
for t, b in enumerate(text):
    if prev is not None:
        recs_prev, logits = prev
        ce = cross_entropy(logits, b)
        ces.append(float(ce.data))
        for p in model.params.values(): p.tg.clear()
        backward_task(ce, [recs_prev[-1]], "ce")
        for p in model.params.values():
            g = p.tg.get("ce")
            if g is None: continue
            n = np.linalg.norm(g)
            if n > 1.0: g = g / n                       # clip per-parameter
            p.data = p.data - 0.05 * lr_scale * g
    x = model.encode_byte(b)
    recs, st = stack_step(model, x, st, tgt[b])
    prev = (recs, byte_logits(model, recs[-1].pred(model, L, 1)))
dt = time.time() - t0
k = max(len(ces)//5, 1)
print(f"{len(text)/dt:.0f} byte/detik")
print("CE per seperlima:", [round(float(np.mean(ces[i:i+k])),3) for i in range(0, len(ces), k)][:5], "| ln256 = 5.545")
