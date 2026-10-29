"""Training arguments and per-epoch history stored in Ultralytics checkpoints."""
import sys
import torch

DEFAULTS_SHOWN = ("model", "data", "epochs", "patience", "batch", "imgsz", "optimizer", "lr0", "lrf",
                  "momentum", "weight_decay", "warmup_epochs", "cos_lr", "close_mosaic", "mosaic",
                  "mixup", "copy_paste", "scale", "translate", "degrees", "fliplr", "flipud",
                  "hsv_h", "hsv_s", "hsv_v", "erasing", "seed", "fraction", "freeze", "amp", "name")


def main():
    for path in sys.argv[1:]:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        a = ck.get("train_args", {})
        print(f"== {path}")
        print("   epoch:", ck.get("epoch"), " best_fitness:", ck.get("best_fitness"), " date:", ck.get("date"),
              " version:", ck.get("version"))
        print("   " + "  ".join(f"{k}={a.get(k)}" for k in DEFAULTS_SHOWN))
        m = ck.get("train_metrics") or {}
        if m:
            print("   final metrics:", {k: round(float(v), 4) for k, v in m.items()})
        r = ck.get("train_results") or {}
        if r:
            keys = list(r.keys())
            print("   history columns:", keys)
            ep = r.get("epoch", [])
            for i in range(len(ep)):
                row = {k: r[k][i] for k in keys if isinstance(r[k], list) and i < len(r[k])}
                print("    ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
                              if k in ("epoch", "train/box_loss", "train/cls_loss", "val/box_loss", "val/cls_loss",
                                       "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)",
                                       "metrics/mAP50-95(B)")})


if __name__ == "__main__":
    main()
