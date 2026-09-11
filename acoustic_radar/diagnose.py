#!/usr/bin/env python3
"""
diagnose.py — Measurements this project could previously only guess at.

    python diagnose.py all                 # everything below
    python diagnose.py xvf                 # xvf_host.py subprocess cost
    python diagnose.py beams --samples 60  # what the DSP actually reports
    python diagnose.py cpu --seconds 20    # per-thread CPU of a running main.py
    python diagnose.py thermal

═══════════════════════════════════════════════════════════════════
WHY THIS FILE EXISTS
═══════════════════════════════════════════════════════════════════

Three numbers this system's behaviour depends on were, until now, taken from
a source comment rather than from the machine:

  1. how long one `xvf_host.py` invocation really takes. doa.HardwareDOA
     spawns one every 0.35 s for the entire session, so if that figure is
     wrong the estimate of how much CPU the DOA poller steals from the camera
     and the Hailo thread is wrong too;

  2. what AEC_AZIMUTH_VALUES actually contains over time. `select_azimuth`
     resolves a 2-vs-2 tie by preferring the TIGHTER cluster, and a beam value
     that is simply repeated has a spread of exactly zero — so a duplicate
     wins every tie regardless of where it points. Whether a duplicate means
     "two beams agree" or "this beam is inactive" is not documented anywhere
     in this repository and cannot be reasoned out; it has to be observed;

  3. which thread on a running station is actually consuming the CPU.

Everything printed here is measured on the machine it runs on. Nothing is
estimated, and where a measurement cannot be taken the tool says so instead
of substituting a plausible number.

Standard library only — this must run on a Pi with nothing extra installed.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def _hr(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def _stats(values):
    """(n, mean, p50, p95, min, max) — or None for an empty series."""
    if not values:
        return None
    data = sorted(values)
    n = len(data)
    return (n, sum(data) / n, data[n // 2],
            data[min(n - 1, int(n * 0.95))], data[0], data[-1])


def _print_stats(label: str, values, unit: str = "ms") -> None:
    s = _stats(values)
    if s is None:
        print(f"   {label:<26} NOT MEASURED (no samples)")
        return
    n, mean, p50, p95, lo, hi = s
    print(f"   {label:<26} mean {mean:8.1f}  p50 {p50:8.1f}  p95 {p95:8.1f}  "
          f"min {lo:8.1f}  max {hi:8.1f} {unit}  (n={n})")


# ═══════════════════════════════════════════════════════════════
#  1. xvf_host.py subprocess cost
# ═══════════════════════════════════════════════════════════════

def find_script(explicit: str | None) -> Path | None:
    if explicit:
        p = Path(explicit).expanduser()
        return p if p.exists() else None
    try:
        from doa import find_xvf_host
    except Exception as exc:
        print(f"   could not import doa.find_xvf_host: {exc}")
        return None
    return find_xvf_host()


def cmd_xvf(args) -> int:
    _hr("1. xvf_host.py SUBPROCESS COST — the real number, not the comment")

    script = find_script(args.script)
    if script is None:
        print("   xvf_host.py NOT FOUND.")
        print("   Pass it explicitly:  python diagnose.py xvf --script /path/to/xvf_host.py")
        print("   or set XVF_HOST_PATH. TEST NOT PERFORMED — no result is being invented.")
        return 1
    print(f"   script : {script}")
    print(f"   cwd    : {script.parent}   (doa.HardwareDOA runs it from here;")
    print(f"            xvf_host.py reads its command maps relative to its own")
    print(f"            directory and fails when started from elsewhere)")
    print(f"   python : {sys.executable}")
    print()

    # A bare interpreter start, for reference: it separates "python is slow to
    # boot" from "this particular script is slow", which decide different fixes.
    baseline = []
    for _ in range(min(args.samples, 10)):
        t0 = time.monotonic()
        try:
            subprocess.run([sys.executable, "-c", "pass"],
                           capture_output=True, timeout=10)
        except Exception as exc:
            print(f"   bare interpreter start failed: {exc}")
            break
        baseline.append((time.monotonic() - t0) * 1000.0)
    _print_stats("bare python -c pass", baseline)

    runtimes, failures, outputs = [], 0, []
    for i in range(args.samples):
        t0 = time.monotonic()
        try:
            out = subprocess.run(
                [sys.executable, str(script), "AEC_AZIMUTH_VALUES"],
                capture_output=True, text=True, timeout=args.timeout,
                cwd=str(script.parent))
        except subprocess.TimeoutExpired:
            failures += 1
            continue
        except OSError as exc:
            print(f"   could not run it at all: {exc}")
            return 1
        dt = (time.monotonic() - t0) * 1000.0
        runtimes.append(dt)
        outputs.append((out.stdout or "") + "\n" + (out.stderr or ""))
        if i == 0:
            print("   first invocation, raw output:")
            for line in ((out.stdout or "") + (out.stderr or "")).splitlines()[:6]:
                print(f"      | {line}")
            print()

    _print_stats("xvf_host AEC_AZIMUTH_VALUES", runtimes)
    if failures:
        print(f"   timeouts: {failures}/{args.samples}")

    s = _stats(runtimes)
    if s is not None:
        mean = s[1]
        interval = 0.35        # doa.HardwareDOA.interval
        period_ms = interval * 1000.0 + mean
        duty = mean / period_ms
        print()
        print("   WHAT THIS COSTS THE STATION")
        print(f"      doa.HardwareDOA polls every {interval:.2f} s and each poll")
        print(f"      blocks for {mean:.0f} ms, so one poller thread is busy")
        print(f"      {duty * 100:.0f}% of the time, {1000.0 / period_ms:.1f} times a second,")
        print(f"      for the ENTIRE session — including while nothing is detected.")
        print(f"      On a 4-core Pi 5 that is ~{duty * 25:.0f}% of total CPU.")
        print()
        print("      Note this is wall time, and subprocess.run RELEASES the GIL")
        print("      while waiting, so the cost is CPU contention (the spawned")
        print("      interpreter's own startup work), not GIL blocking.")
    return 0


# ═══════════════════════════════════════════════════════════════
#  2. What the DSP actually reports
# ═══════════════════════════════════════════════════════════════

#: Below this many parsed samples nothing can be concluded about a beam.
BEAM_MIN_SAMPLES = 100


def beam_verdict(n_samples: int, changes: int,
                 scene_changed: bool,
                 min_samples: int = BEAM_MIN_SAMPLES) -> str:
    """
    What a beam's behaviour over one run actually establishes.

    ═══════════════════════════════════════════════════════════════
    ⚠️ FORENSIC REVIEW DEFECT D6 — "STEADY" IS NOT "FROZEN"
    ═══════════════════════════════════════════════════════════════

    The previous version printed `FROZEN (no change in the whole run)`
    whenever a beam never changed across >= 100 samples. But a beam that is
    working perfectly and watching a source that never moved ALSO never
    changes. A sample count cannot separate those two — no matter how long
    the run is, a static scene produces a static reading.

    Only ONE thing distinguishes them: deliberately changing the acoustic
    scene and seeing whether the beam follows. The tool cannot observe
    that, so the operator asserts it with `--scene-changed`, and without
    that assertion a zero-change beam is reported as NOT PROVEN rather than
    condemned.

    This matters because the whole point of finding H4 was that the
    original "beams 0 and 1 are frozen" conclusion outran its evidence.
    Reproducing that error in the tool built to correct it would be
    particularly poor.

    Returns a human-readable verdict string beginning with one of:
        INSUFFICIENT / NOT PROVEN / FROZEN / RESPONDED / NEARLY STATIC
    """
    if n_samples < 2:
        return f"INSUFFICIENT — {n_samples} sample(s), nothing to compare"
    if n_samples < min_samples:
        return (f"INSUFFICIENT — only {n_samples} samples (need "
                f"{min_samples}); {changes} change(s) seen")

    if changes == 0:
        if not scene_changed:
            return (f"NOT PROVEN — STATIC SCENE. No change in {n_samples} "
                    f"samples, but the source was not reported as moved, so "
                    f"a healthy beam watching a still source looks exactly "
                    f"like this. Re-run with --scene-changed after moving "
                    f"or switching off the source.")
        return (f"FROZEN — no change in {n_samples} samples ACROSS A "
                f"DELIBERATE SOURCE CHANGE; the beam is not being updated")

    if scene_changed:
        return (f"RESPONDED — {changes} change(s) in {n_samples} samples, "
                f"including across the source change; the beam is live")
    if changes < n_samples * 0.05:
        return (f"NEARLY STATIC — {changes} change(s) in {n_samples} "
                f"samples; the beam does update, so it is not frozen")
    return f"UPDATING — {changes} change(s) in {n_samples} samples"


def cmd_beams(args) -> int:
    _hr("2. AEC_AZIMUTH_VALUES OVER TIME — is a duplicated beam meaningful?")

    script = find_script(args.script)
    if script is None:
        print("   xvf_host.py NOT FOUND — TEST NOT PERFORMED.")
        return 1

    try:
        from doa import parse_azimuth_values, azimuths_to_degrees, select_azimuth
    except Exception as exc:
        print(f"   could not import doa: {exc}")
        print("   (this part needs numpy; run it in the station's virtualenv)")
        return 1

    print(f"   Put a STEADY sound source at a KNOWN direction and leave it there")
    print(f"   for the whole run. {args.samples} samples, ~{args.interval:.2f} s apart.")
    print()

    rows, ties, dup_rows = [], 0, 0
    per_beam = [[], [], [], []]
    for _ in range(args.samples):
        try:
            out = subprocess.run(
                [sys.executable, str(script), "AEC_AZIMUTH_VALUES"],
                capture_output=True, text=True, timeout=args.timeout,
                cwd=str(script.parent))
        except Exception:
            continue
        text = (out.stdout or "") + "\n" + (out.stderr or "")
        degrees = azimuths_to_degrees(parse_azimuth_values(text))
        if not degrees:
            continue
        angle, conf, amb = select_azimuth(degrees)
        rows.append((degrees, angle, conf, amb))
        if amb:
            ties += 1
        rounded = [round(d, 1) for d in degrees]
        if len(set(rounded)) < len(rounded):
            dup_rows += 1
        for i, d in enumerate(degrees[:4]):
            per_beam[i].append(d)
        time.sleep(args.interval)

    if not rows:
        print("   NO SAMPLES PARSED — the DSP returned nothing usable.")
        return 1

    print("   sample  beams                                  selected  conf  tie")
    for degrees, angle, conf, amb in rows[:args.show]:
        beams = " ".join(f"{d:6.1f}" for d in degrees)
        print(f"      {beams:<38} {angle:7.1f}  {conf:4.0%}  "
              f"{'YES' if amb else '-'}")
    if len(rows) > args.show:
        print(f"      ... {len(rows) - args.show} more")

    print()
    print(f"   samples parsed                : {len(rows)}")
    print(f"   rows containing a DUPLICATE   : {dup_rows} "
          f"({dup_rows / len(rows):.0%})")
    print(f"   rows resolved as an AMBIGUOUS : {ties} "
          f"({ties / len(rows):.0%}) 2-vs-2 tie")
    # ═══════════════════════════════════════════════════════════
    #  PER-BEAM STATISTICS — "frozen" vs "merely steady"
    # ═══════════════════════════════════════════════════════════
    #
    # ⚠️ THIS SECTION WAS EXPANDED BECAUSE THE PREVIOUS CONCLUSION WAS NOT
    # SUPPORTED BY ITS OWN EVIDENCE (audit finding H4). An earlier run of
    # 20 samples over ~7 s showed beams 0 and 1 with 1 and 2 distinct
    # values, and that was written up as "two beams are frozen". Twenty
    # samples cannot separate two very different situations:
    #
    #   FROZEN    the DSP is not updating that beam at all — its value is
    #             a constant, and feeding it to the selector manufactures
    #             agreement out of nothing (a constant always has spread
    #             0.0, so it always looks like the tightest cluster).
    #   STEADY    the beam IS updating and is correctly reporting a source
    #             that genuinely did not move. That is the beam WORKING.
    #
    # The difference is not in the spread, it is in whether the value ever
    # CHANGES over a long enough window, and whether it changes when the
    # acoustic scene changes. So the run must be long, and it must include
    # a period with the source moved or switched off. Mean and range alone
    # cannot answer it, which is why they are no longer all that is shown.
    def _stats(vals):
        n = len(vals)
        mean = sum(vals) / n
        var = sum((v - mean) ** 2 for v in vals) / n
        # Longest run of BIT-IDENTICAL consecutive readings. This, not the
        # spread, is the signature of a beam that is not being updated.
        longest = run = 1
        changes = 0
        for a, b in zip(vals, vals[1:]):
            if a == b:
                run += 1
                longest = max(longest, run)
            else:
                run = 1
                changes += 1
        return mean, var, var ** 0.5, longest, changes

    print()
    print("   PER-BEAM BEHAVIOUR OVER THE RUN")
    print("   (spread says how far it moved; 'changes' and 'max run' say")
    print("    whether it is being UPDATED at all — that is the real test)")
    print()
    print("      beam    n  uniq     mean      min      max   spread"
          "    stdev      var  changes  max-run")
    verdicts = {}
    for i, vals in enumerate(per_beam):
        if len(vals) < 2:
            print(f"      {i:4d}  {len(vals):3d}   -- not enough samples --")
            verdicts[i] = "NO DATA"
            continue
        uniq = len(set(vals))
        lo, hi = min(vals), max(vals)
        mean, var, sd, longest, changes = _stats(vals)
        print(f"      {i:4d}  {len(vals):3d}  {uniq:4d}  {mean:7.1f}  "
              f"{lo:7.1f}  {hi:7.1f}  {hi - lo:7.1f}  {sd:7.2f}  "
              f"{var:7.2f}  {changes:7d}  {longest:7d}")

        verdicts[i] = beam_verdict(len(vals), changes,
                                   getattr(args, "scene_changed", False))

    print()
    print("   VERDICT PER BEAM")
    for i, verdict in sorted(verdicts.items()):
        print(f"      beam {i}: {verdict}")

    # ⚠️ Printed on EVERY run, not only short ones (defect D6). The scene
    # change is the only thing that separates "frozen" from "steady", so
    # the instruction must not be hidden behind a sample-count condition.
    if len(rows) < BEAM_MIN_SAMPLES:
        print()
        print(f"   ⚠️  ONLY {len(rows)} SAMPLES (need {BEAM_MIN_SAMPLES}).")
        print( "      Re-run with at least:")
        print( "          python diagnose.py beams --samples 200 --interval 0.5")
    if not getattr(args, "scene_changed", False):
        print()
        print( "   ⚠️  --scene-changed WAS NOT GIVEN, so no beam can be")
        print( "      reported FROZEN by this run, however long it was.")
        print( "      A beam that is perfectly healthy but watching a source")
        print( "      that never moved is indistinguishable from a frozen one;")
        print( "      only changing the scene tells them apart. Re-run as:")
        print( "          python diagnose.py beams --samples 200 "
               "--interval 0.5 --scene-changed")
        print( "      and, partway through, MOVE the source or switch it off.")
        print( "      Record what you did and when.")
        print( "      See HARDWARE_TEST_REQUIRED.md, test HW-7.")

    print()
    print("   HOW TO READ THIS")
    print("      • If ONE beam is consistently steady and points at the known")
    print("        direction, set that index as doa_beam_index in")
    print("        radar_calibration.json and the tie-break stops mattering.")
    print("      • If the duplicated value is the one that tracks the source,")
    print("        the current tie-break is doing the right thing by accident.")
    print("      • If the duplicated value is NOT the one that tracks it, the")
    print("        tie-break is systematically choosing the wrong cluster,")
    print("        because a repeated number always has spread 0.0 and always")
    print("        wins. That would be a CONFIRMED bug — but only this")
    print("        measurement can establish it.")
    return 0


# ═══════════════════════════════════════════════════════════════
#  3. Per-thread CPU of a running station
# ═══════════════════════════════════════════════════════════════

def _clock_ticks() -> float:
    try:
        return float(os.sysconf("SC_CLK_TCK"))
    except (ValueError, OSError, AttributeError):
        return 100.0


def _thread_times(pid: int) -> dict:
    """{tid: (name, utime+stime in ticks)} from /proc. Linux only."""
    out = {}
    task = Path(f"/proc/{pid}/task")
    if not task.is_dir():
        return out
    for entry in task.iterdir():
        try:
            stat = (entry / "stat").read_text()
        except OSError:
            continue
        # comm can contain spaces and parentheses; fields after the last ')'.
        close = stat.rfind(")")
        name = stat[stat.find("(") + 1:close]
        fields = stat[close + 2:].split()
        try:
            utime, stime = int(fields[11]), int(fields[12])
        except (IndexError, ValueError):
            continue
        out[entry.name] = (name, utime + stime)
    return out


def cmd_cpu(args) -> int:
    _hr("3. PER-THREAD CPU OF THE RUNNING STATION")

    if not Path("/proc").is_dir():
        print("   /proc is not available — this test only works on Linux.")
        print("   TEST NOT PERFORMED.")
        return 1

    pid = args.pid
    if pid is None:
        try:
            found = subprocess.run(["pgrep", "-f", "python.*main.py"],
                                   capture_output=True, text=True, timeout=5)
            pids = [int(p) for p in found.stdout.split()]
            pids = [p for p in pids if p != os.getpid()]
            pid = pids[0] if pids else None
        except Exception:
            pid = None
    if pid is None:
        print("   No running main.py found. Start it first:")
        print("      python main.py --headless &")
        print("   then:  python diagnose.py cpu")
        print("   TEST NOT PERFORMED.")
        return 1

    print(f"   pid {pid}, sampling for {args.seconds:.0f} s")
    ticks = _clock_ticks()
    before = _thread_times(pid)
    if not before:
        print(f"   could not read /proc/{pid}/task — is the pid right?")
        return 1
    t0 = time.monotonic()
    time.sleep(args.seconds)
    elapsed = time.monotonic() - t0
    after = _thread_times(pid)

    rows = []
    for tid, (name, total) in after.items():
        prev = before.get(tid)
        if prev is None:
            continue
        delta_s = (total - prev[1]) / ticks
        rows.append((delta_s / elapsed * 100.0, name, tid))
    rows.sort(reverse=True)

    print()
    print(f"   {'%CPU':>7}  {'thread':<18} tid")
    for pct, name, tid in rows:
        print(f"   {pct:7.1f}  {name:<18} {tid}")
    print(f"   {sum(r[0] for r in rows):7.1f}  TOTAL (100% = one full core)")

    try:
        statm = Path(f"/proc/{pid}/statm").read_text().split()
        rss_mb = int(statm[1]) * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
        print(f"\n   RSS: {rss_mb:.0f} MB")
    except Exception:
        pass
    return 0


# ═══════════════════════════════════════════════════════════════
#  4. Thermal / throttling
# ═══════════════════════════════════════════════════════════════

def cmd_thermal(_args) -> int:
    _hr("4. THERMAL AND THROTTLING")
    zone = Path("/sys/class/thermal/thermal_zone0/temp")
    if zone.exists():
        try:
            print(f"   SoC temperature: {int(zone.read_text()) / 1000.0:.1f} C")
        except (OSError, ValueError) as exc:
            print(f"   could not read {zone}: {exc}")
    else:
        print("   /sys/class/thermal/thermal_zone0/temp not present — "
              "TEST NOT PERFORMED (not a Pi?)")

    try:
        out = subprocess.run(["vcgencmd", "get_throttled"],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0:
            value = out.stdout.strip()
            print(f"   {value}")
            print("      0x0 = never throttled. Bit 0 = under-voltage NOW,")
            print("      bit 1 = frequency capped, bit 2 = throttled,")
            print("      bits 16-18 = the same conditions since boot.")
            print("      Anything non-zero makes every FPS number below the")
            print("      hardware's real capability, and no software change")
            print("      will fix it — check the power supply and cooling.")
        else:
            print("   vcgencmd present but returned an error.")
    except FileNotFoundError:
        print("   vcgencmd not found — throttling state NOT MEASURED.")
    except Exception as exc:
        print(f"   vcgencmd failed: {exc}")
    return 0


# ═══════════════════════════════════════════════════════════════

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure what this project used to assume.")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("xvf", help="real xvf_host.py subprocess cost")
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--script", default=None)
    p.set_defaults(func=cmd_xvf)

    p = sub.add_parser("beams", help="what AEC_AZIMUTH_VALUES really reports")
    # ⚠️ Raised from 40 to 200 (audit finding H4). At 0.35 s per sample the
    # old default covered ~14 s, which is far too short to distinguish a
    # beam that is frozen from one correctly reporting a source that did
    # not happen to move. 200 samples is ~70 s and leaves room to move the
    # source partway through, which is what actually settles the question.
    p.add_argument("--samples", type=int, default=200)
    # ⚠️ The operator asserts that the acoustic scene was deliberately
    # changed mid-run (source moved or switched off). Without it the tool
    # will not call any beam FROZEN — see beam_verdict, defect D6.
    p.add_argument("--scene-changed", dest="scene_changed",
                   action="store_true",
                   help="you MOVED or switched off the source partway "
                        "through this run; required before any beam can be "
                        "reported FROZEN")
    p.add_argument("--interval", type=float, default=0.35)
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--show", type=int, default=20)
    p.add_argument("--script", default=None)
    p.set_defaults(func=cmd_beams)

    p = sub.add_parser("cpu", help="per-thread CPU of a running main.py")
    p.add_argument("--seconds", type=float, default=15.0)
    p.add_argument("--pid", type=int, default=None)
    p.set_defaults(func=cmd_cpu)

    p = sub.add_parser("thermal", help="SoC temperature and throttling")
    p.set_defaults(func=cmd_thermal)

    p = sub.add_parser("all", help="every test above")
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--interval", type=float, default=0.35)
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--show", type=int, default=20)
    p.add_argument("--script", default=None)
    p.add_argument("--seconds", type=float, default=15.0)
    p.add_argument("--pid", type=int, default=None)
    p.set_defaults(func=None)

    args = parser.parse_args(argv)
    if args.cmd is None:
        parser.print_help()
        return 2
    if args.cmd == "all":
        rc = 0
        for fn in (cmd_thermal, cmd_xvf, cmd_beams, cmd_cpu):
            try:
                rc |= fn(args)
            except KeyboardInterrupt:
                print("\ninterrupted")
                return 130
        return rc
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
