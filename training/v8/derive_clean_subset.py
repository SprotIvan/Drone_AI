#!/usr/bin/env python3
"""
derive_clean_subset.py — test_inet minus the images any ancestor of the models
trained on (leak_check.py vs the v2 / v5 / v6 train sets: 194 of 826 are
byte-identical to v2 TRAIN images - ft_v2 trained on them, and ft_v5, ft_v7 and
ft_v8 all descend from ft_v2).

    python derive_clean_subset.py make LEAK.json            -> testsets/test_inet_clean/ (images + labels)
    python derive_clean_subset.py add PREDS.json [...]      -> adds set "test_inet_clean" to each eval_v8 preds
                                                               file, filtered from its "test_inet" set
"""
import json
import shutil
import sys
from pathlib import Path

TS = Path(r"C:\Github\drone_training\v8\testsets")


def main():
    cmd = sys.argv[1]
    dst = TS / "test_inet_clean"
    if cmd == "make":
        leak = json.loads(Path(sys.argv[2]).read_text())["test_inet"]
        clean = sorted(Path(k).name for k, v in leak.items() if not any(v.values()))
        seen = sorted(Path(k).name for k, v in leak.items() if any(v.values()))
        if dst.exists():
            sys.exit(f"{dst} exists")
        (dst / "images").mkdir(parents=True)
        (dst / "labels").mkdir(parents=True)
        for n in clean:
            shutil.copy2(TS / "test_inet" / "images" / n, dst / "images" / n)
            shutil.copy2(TS / "test_inet" / "labels" / (Path(n).stem + ".txt"), dst / "labels" / (Path(n).stem + ".txt"))
        (dst / "excluded_seen_by_ft_v2.json").write_text(json.dumps(
            {"reason": "byte-identical (md5) to a drone_dataset_v2 TRAIN image (ft_v2 trained on it; "
                       "ft_v5/ft_v7/ft_v8 descend from ft_v2)", "images": seen}, indent=0))
        print(f"test_inet_clean: {len(clean)} images (excluded {len(seen)})")
    else:
        clean = {p.name for p in (dst / "images").iterdir()}
        for f in sys.argv[2:]:
            d = json.loads(Path(f).read_text())
            src = d["sets"]["test_inet"]
            d["sets"]["test_inet_clean"] = {k: v for k, v in src.items() if Path(k).name in clean}
            Path(f).write_text(json.dumps(d))
            print(f"{Path(f).name}: test_inet_clean {len(d['sets']['test_inet_clean'])} of {len(src)}")


if __name__ == "__main__":
    main()
