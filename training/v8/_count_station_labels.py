import json
from collections import Counter, defaultdict
from pathlib import Path
S = Path(r"C:\Users\Nazar\AppData\Local\Temp\claude\c--Github-THIS-IS-CAMERA-ANDAU\1af7577f-dcd9-409f-9d28-467db2513d82\scratchpad")
frames = json.loads((S / "pi_v57" / "frames_unseen.json").read_text())
objs = json.loads((S / "sess_scene" / "sess_objects.json").read_text())
lab = json.loads((S / "sess_scene" / "sess_labels.json").read_text())
TRAIN = {"frames_20260919_140445", "frames_20260919_141424", "frames_20260919_160325", "frames_20260919_160513", "frames_20260919_180118"}
per = defaultdict(lambda: defaultdict(set))
for o in objs:
    s = o["session"]
    per[s][lab[str(o["id"])]].add(o["frame"])
nfr = Counter(fr["session"] for fr in frames)
for s in sorted(nfr):
    d, f, u = per[s]["D"], per[s]["F"], per[s]["U"]
    fonly = f - d - u
    print(f"{'TRAIN' if s in TRAIN else 'test '} {s}: clean frames {nfr[s]:4d} | frames with D {len(d):3d} | with F only {len(fonly):3d} | with U {len(u):3d} | objects D {sum(1 for o in objs if o['session']==s and lab[str(o['id'])]=='D')} F {sum(1 for o in objs if o['session']==s and lab[str(o['id'])]=='F')}")
