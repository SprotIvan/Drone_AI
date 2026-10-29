import sys
sys.path.insert(0, r"C:\Github\THIS_IS_CAMERA_ANDAU\training\v8")
import station_additions as SA

frames, byf = SA.load()
for a in sys.argv[1:]:
    lo, _, hi = a.partition("-")
    for i in range(int(lo), int(hi or lo) + 1):
        fr = frames[i]
        obs = byf.get(i, [])
        print(i, fr["session"][16:], fr["pub"], "objects:", len(obs))
        for o in obs:
            (x, y, w, h), sc, m = SA.best_box(o)
            print(f"   id {o['id']} {o['lab']}  best {m} {sc}  box {x},{y},{w},{h}  n_models {len(o['boxes'])}")
