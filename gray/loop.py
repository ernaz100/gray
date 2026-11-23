#!/usr/bin/env python3
"""The replanning loop, run along recordings: is a mismatch a good moment to plan again?

    python -m gray.loop --run <runs>/joint-v1

The loop. At a moment "now" the model has the last second of what happened (head poses, image
features, body motion) and the text. ``plan`` makes

    the plan         head path, then features, then body motion for the next 5 s (``joint.generate``)
    the prediction   the expected image feature at every frame of that time (``joint.expected``)

While the plan runs, every new camera frame is compared with the prediction for it. The mismatch
is their mean squared difference (standardized features, 1.0 is the average frame), averaged
over 0.2 s. When the trigger fires, the loop plans again from the new now, with the last second
again given, so the motion continues from where the body is.

On a recording the camera follows the person, not the plan. So the prediction is made along the
head path the person really took (as agreed, see DESIGN.md): the mismatch then says "the world is
not what I expected" and not "I left my path". How good a plan still is can be measured, though:
by how far its head and pelvis are from the recorded ones at each moment.

To compare triggers without running the model once per trigger, the plan and the prediction are
computed for every possible now on a 0.2 s grid (``tables``). Each trigger is then replayed on
these tables (``replay``), at many settings, which gives its error at any number of replans:

    fixed       plan again every k steps
    random      plan again at random moments
    mismatch    plan again when the mismatch exceeds a threshold (the agreed plain threshold)
    relative    the same, with the mismatch divided by what is usual at this age of the plan
    oracle      plan again when the plan's head is further than a threshold from the recorded one;
                not available to a deployed loop, shown as the best a trigger could do

Every trigger waits at least 0.4 s between plans and plans again after 5 s at the latest. The
windows are cut to 6 s (1 s given, 5 s planned) so that any now on the grid lies in a cached
window; the model was trained on 8 s windows, so the expected head path and the mismatch are
also computed on whole windows and compared on the same frames.
"""

from __future__ import annotations

import argparse
import json
import resource
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from gray.features import RATE
from gray.joint import FEATURE, HEAD, PART_OF, ROOT, Windows, collate, expected, generate, load, raw_tokens, seconds_from, to_device

HISTORY = 15  # tokens given: 1 s
STRIDE = 3  # tokens per grid step: 0.2 s
AGES = 25  # grid steps a plan lasts: 5 s
LENGTH = HISTORY + AGES * STRIDE
SHIFTS = 30 // STRIDE  # grid steps between cached windows
MIN_GAP = 2  # grid steps between plans
STEP_S = STRIDE / RATE
RATES = (15, 20, 30, 45, 60)  # replans per minute the triggers are compared at


@torch.no_grad()
def plan(model, x1: torch.Tensor, now: torch.Tensor, text: torch.Tensor, text_valid: torch.Tensor, steps: int,
         generator: torch.Generator, amp: bool, head_path: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """One planning step of the loop for a batch of standardized windows (B, L, 429) whose tokens up to ``now`` are given.

    Returns the plan (B, L, 429) and the expected features (B, L, 384), both standardized. The
    features are predicted along ``head_path`` (B, L, 9, standardized) if it is given, as on a
    recording, and along the planned head path otherwise, as in a deployed loop.
    """
    seconds = seconds_from(now, x1.shape[1])
    known = (seconds <= 0)[..., None].repeat(1, 1, 3)
    planned = generate(model, x1, known, seconds, text, text_valid, steps=steps, generator=generator, amp=amp)
    along = torch.cat([planned[..., :9] if head_path is None else head_path, x1[..., 9:]], dim=-1)
    known_head = known.clone()
    known_head[..., HEAD] = True
    predicted = expected(model, along, known_head, seconds, text, text_valid, generator=generator, amp=amp)
    return planned, predicted[..., PART_OF.to(x1.device) == FEATURE]


@torch.no_grad()
def tables(model, mean: torch.Tensor, std: torch.Tensor, cache: Path, sequence: str, device: str, steps: int, batch_size: int,
           images: Path | None = None) -> dict[str, np.ndarray] | None:
    """For every now on the grid of a recording: mismatch, head error and pelvis error (cm) of the plan made then, by age (G, 25)."""
    starts = np.load(cache / sequence / "starts.npy")
    if len(starts) < 2 or (np.diff(starts) != 60).any():
        return None  # the grid needs windows every 2 s without gaps
    generator = torch.Generator(device=device).manual_seed(0)
    amp = device == "cuda"
    rows = (torch.arange(SHIFTS, device=device) * STRIDE)[:, None] + torch.arange(LENGTH, device=device)[None]  # (10, 90)
    out = {name: [] for name in ("mismatch", "head_cm", "pelvis_cm")}
    check = np.zeros(4)
    by_age = lambda value: value[:, HISTORY:].reshape(-1, AGES, STRIDE).mean(-1).cpu().numpy()
    loader = DataLoader(Windows(cache, [(sequence, w) for w in range(len(starts))], images), batch_size=batch_size, collate_fn=collate)
    for batch in loader:
        batch = to_device(batch, device)
        size = batch["head"].shape[0]
        cut = {key: batch[key][:, rows].flatten(0, 1) for key in ("head", "image", "latents", "root")}  # (B * 10, 90, ...)
        text, text_valid = (batch[key].repeat_interleave(SHIFTS, dim=0) for key in ("text", "text_valid"))
        now = torch.full((size * SHIFTS,), HISTORY - 1, device=device)
        raw = raw_tokens(cut, now)
        x1 = (raw - mean) / std
        planned, predicted = plan(model, x1, now, text, text_valid, steps, generator, amp, head_path=x1[..., :9])
        planned = planned * std + mean
        feature = PART_OF.to(device) == FEATURE
        mismatch = ((predicted - x1[..., feature]) ** 2).mean(-1)
        out["mismatch"].append(by_age(mismatch))
        out["head_cm"].append(by_age((planned[..., :3] - raw[..., :3]).norm(dim=-1) * 100))
        out["pelvis_cm"].append(by_age((planned[..., -ROOT:-2] - raw[..., -ROOT:-2]).norm(dim=-1) * 100))

        # The same now (token 14 of the cached window) in a cut and in a whole window: mismatch and expected head path.
        def expected_head_and_mismatch(values: dict, text_: tuple) -> tuple[torch.Tensor, torch.Tensor]:
            at = torch.full((values["head"].shape[0],), HISTORY - 1, device=device)
            raw_ = raw_tokens(values, at)
            x = (raw_ - mean) / std
            seconds = seconds_from(at, x.shape[1])
            known = (seconds <= 0)[..., None].repeat(1, 1, 3)
            path = (expected(model, x, known, seconds, *text_, generator=generator, amp=amp) * std + mean)[..., :3]
            known[..., HEAD] = True
            seen = expected(model, x, known, seconds, *text_, generator=generator, amp=amp)[..., feature]
            return ((path - raw_[..., :3]).norm(dim=-1) * 100)[:, HISTORY:LENGTH], ((seen - x[..., feature]) ** 2).mean(-1)[:, HISTORY:LENGTH]

        first = {key: value[::SHIFTS] for key, value in cut.items()}
        whole = {key: batch[key] for key in cut}
        pair = (batch["text"], batch["text_valid"])
        results = [*expected_head_and_mismatch(first, pair), *expected_head_and_mismatch(whole, pair)]
        check += np.array([float(r.mean()) * size for r in results])
    result = {name: np.concatenate(value) for name, value in out.items()}
    result["check"] = check / len(starts)  # expected head cm and mismatch: cut, then whole
    return result


def replay(table: dict[str, np.ndarray], trigger) -> tuple[int, int, float, float]:
    """Run one trigger along a recording. Returns replans, grid steps, and the summed head and pelvis error of the active plan."""
    count = len(table["mismatch"])
    made, replans, head, pelvis = 0, 1, 0.0, 0.0
    for g in range(1, count):
        age = g - made  # in grid steps; column age - 1
        head += table["head_cm"][made, age - 1]
        pelvis += table["pelvis_cm"][made, age - 1]
        if age >= AGES or trigger(made, age):
            made, replans = g, replans + 1
    return replans, count - 1, head, pelvis


def curves(all_tables: dict[str, dict[str, np.ndarray]], seed: int = 0) -> dict[str, list[dict]]:
    """Every trigger at many settings, pooled over the recordings: replans per minute, head and pelvis error."""
    mismatch = np.concatenate([t["mismatch"] for t in all_tables.values()])
    usual = np.median(mismatch, axis=0)  # by age
    levels = np.linspace(0.05, 0.995, 30)
    rng = np.random.default_rng(seed)
    families = {
        "fixed": [(k, lambda t, k=k: lambda made, age: age >= k) for k in range(1, AGES + 1)],
        "random": [(p, lambda t, p=p: lambda made, age: age >= MIN_GAP and rng.random() < p) for p in np.geomspace(0.02, 0.6, 20)],
        "mismatch": [(q, lambda t, q=q: lambda made, age: age >= MIN_GAP and t["mismatch"][made, age - 1] > q) for q in np.quantile(mismatch, levels)],
        "relative": [(q, lambda t, q=q: lambda made, age: age >= MIN_GAP and t["mismatch"][made, age - 1] > q * usual[age - 1])
                     for q in np.quantile(mismatch / usual, levels)],
        "oracle": [(q, lambda t, q=q: lambda made, age: age >= MIN_GAP and t["head_cm"][made, age - 1] > q) for q in np.geomspace(3, 300, 30)],
    }
    result = {}
    for name, settings in families.items():
        result[name] = []
        for setting, make in settings:
            totals = np.sum([replay(t, make(t)) for t in all_tables.values()], axis=0)
            result[name].append({"setting": float(setting), "replans_per_minute": float(totals[0] / (totals[1] * STEP_S / 60)),
                                 "head_cm": float(totals[2] / totals[1]), "pelvis_cm": float(totals[3] / totals[1])})
    return result


def at_rates(points: list[dict], key: str) -> list[float]:
    """The error of one trigger at the replans per minute of RATES, by linear interpolation; nan outside its range."""
    points = sorted(points, key=lambda p: p["replans_per_minute"])
    rate, value = [p["replans_per_minute"] for p in points], [p[key] for p in points]
    return [float(np.interp(r, rate, value)) if rate[0] <= r <= rate[-1] else float("nan") for r in RATES]


def pick(cache: Path, config: dict, per_location: int) -> list[str]:
    """Recordings of the held-out locations the run was evaluated on, the first few of each by name."""
    sequences = json.loads((cache / "index.json").read_text())["sequences"]
    locations = sorted({m["location"] for m in sequences})
    every = max(2, round(1 / float(config["holdout"])))
    picked = []
    for location in locations[every - 1 :: every]:
        here = sorted(m["sequence_id"] for m in sequences if m["location"] == location and m["clock_gyro_correlation"] >= float(config["min_clock"]))
        scripts_seen, chosen = set(), []
        for name in here:  # different scripts first
            script = next(m.get("script") for m in sequences if m["sequence_id"] == name)
            if script not in scripts_seen and len(chosen) < per_location:
                scripts_seen.add(script)
                chosen.append(name)
        picked += chosen
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("sequences", nargs="*", help="recordings; default: some of each held-out location")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=None)
    parser.add_argument("--per-location", type=int, default=2)
    parser.add_argument("--steps", type=int, default=10, help="sampling steps per part")
    parser.add_argument("--batch-size", type=int, default=48, help="cached windows per batch; each gives 10 plans")
    args = parser.parse_args()

    if torch.version.hip:
        torch.backends.cuda.preferred_blas_library("cublas")
    hard = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, mean, std, config = load(args.run, device)
    cache = args.cache or Path(config["cache"])
    images = Path(config["images"]) if config.get("images") else None  # the image features the run was trained on
    out = args.run / "loop"
    out.mkdir(exist_ok=True)

    all_tables = {}
    for name in args.sequences or pick(cache, config, args.per_location):
        table = tables(model, mean, std, cache, name, device, args.steps, args.batch_size, images)
        if table is None:
            print(f"{name}: skipped, its windows have gaps", flush=True)
            continue
        np.savez_compressed(out / f"{name}.npz", **table)
        all_tables[name] = table
        c = table["check"]
        print(f"{name}: {len(table['mismatch']) * STEP_S / 60:.1f} min; mismatch at 1 s {np.median(table['mismatch'][:, 4]):.2f}, at 5 s "
              f"{np.median(table['mismatch'][:, -1]):.2f}; cut against whole windows: expected head {c[0]:.1f} and {c[2]:.1f} cm, "
              f"mismatch {c[1]:.3f} and {c[3]:.3f}", flush=True)

    result = curves(all_tables)
    minutes = sum(len(t["mismatch"]) for t in all_tables.values()) * STEP_S / 60
    print(f"\n{len(all_tables)} recordings, {minutes:.0f} minutes. Error of the plan in use, by replans per minute", flush=True)
    summary = {}
    for key, label in (("head_cm", "head position, cm"), ("pelvis_cm", "pelvis position, cm")):
        print(f"\n  {label:<22}" + "".join(f"{r:>9d}" for r in RATES))
        summary[key] = {name: at_rates(points, key) for name, points in result.items()}
        for name, values in summary[key].items():
            print(f"  {name:<22}" + "".join(f"{v:9.1f}" for v in values), flush=True)
    (out / "loop.json").write_text(json.dumps({"rates": RATES, "summary": summary, "curves": result, "recordings": list(all_tables)}, indent=1))
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
