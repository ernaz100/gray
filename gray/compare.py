#!/usr/bin/env python3
"""The comparator on moments: what will the camera show along the path, and how sure is that?

    python -m gray.compare --save-dir <runs>/compare-v1 --amp

Jeffrey Gray's comparator predicts what the senses will report and raises an alarm when they report
something else. Here the prediction is the image feature of every frame of the next 5 s (the
DINOv3 class token of the pinhole view, gray/pinhole.py; standardized, so 1.0 is the spread of
an average frame), given

    past      the head poses and image features of up to 30 frames up to now
    path      the head poses of the 75 frames after now. At run time these are the poses the head really
              takes, so a mismatch means the world differs, not that the plan was left
    text      the sentence that begins at now
    patches   the 7 x 7 grid of image features of the frame at now

Besides the feature the model predicts, per value, how uncertain it is (a variance; trained
with the Gaussian likelihood). A tracking filter judges a mismatch relative to that: a large
error after turning into a room that was not in view is expected and no alarm, the same error in
a place that was in view is. So the comparator's signal is

    mismatch    the squared error of a frame's prediction, averaged over the 384 values
    surprise    the squared error of each value divided by its predicted variance, averaged: about 1 when
                the world is as expected, whatever the head did

Reported at every evaluation on held-out moments: the mismatch, against copying the last seen
frame; the surprise (1 is calibrated); the rank correlation of the predicted uncertainty with
the actual error; and mismatch and surprise by how far the head has turned since now.
gray/alarm.py tests whether the surprise notices a changed world.

``--revisits`` trains with revisits as targets: half of the frames to come are replaced, where possible, by a frame of
the same recording that shows the same place from nearly the same pose at least 10 s later (gray/revisits.py). The
room is unchanged between the two; hands, people and what is being handled are not. The predicted variance then has to
cover that variation, which a fixed floor added afterwards did only roughly. On the true frames the surprise of such a
model is below 1.

``--target grid`` predicts, in place of the one vector per frame, the 7 x 7 image grid of every fifth frame (15 frames
in the 5 s), each cell reduced to 64 numbers by a principal component analysis of the grid cells. The class token
changes with everything in the image; with a grid the comparison can be restricted to the cells where the path goes
(``corridor``), so that hands at the bottom of the image or a person at the side do not count.

Saved in the run folder: config.json, log.jsonl, last.pt and best.pt, the state with the lowest held-out loss (each with
the mean and deviation of poses and features). ``load`` takes best.pt.
"""

from __future__ import annotations

import argparse
import json
import math
import resource
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from gray import frames
from gray.features import CACHE, RATE, time_embedding
from gray.head import NOW, anchored
from gray.joint import to_device
from gray.moments import FUTURE, HISTORY, Moments, collate

TURNS = (0, 15, 45, 90, 181)  # degrees
EVERY = 5  # with --target grid, every fifth frame after now is predicted
KEPT = list(range(EVERY - 1, FUTURE, EVERY))  # their places among the 75 frames after now
CELLS, PARTS = 49, 64


def corridor(pose: torch.Tensor, place: int, reach: int = 30) -> torch.Tensor:
    """Which cells of the 7 x 7 image grid of one frame show where the path goes next: a mask (49,).

    ``pose`` (L, 9): head poses in any ground frame (x, height above the floor, z, rotation); ``place``: the frame whose
    image is meant. The path over the following ``reach`` frames (2 s) is drawn into that image at floor height,
    0.75 m and 1.5 m; the cells it falls into, and their neighbours, are the corridor. If the path leaves no trace in the
    image (standing still, or the way lies outside the view) the lower middle of the image stands in for "ahead".
    """
    here, ahead = pose[place].double(), pose[place + 1 : place + 1 + reach, :3].double()
    first, second = here[3:6], here[6:9]
    to_ground = torch.stack([first, second, torch.linalg.cross(first, second)])  # the sensor's axes in the ground frame, as columns
    points = torch.cat([torch.stack([ahead[:, 0], torch.full_like(ahead[:, 0], height), ahead[:, 2]], dim=1) for height in (0.0, 0.75, 1.5)])
    sensor = (points - here[:3]) @ to_ground  # x is down in the image, y is left, z looks ahead
    focal = 3.5 / math.tan(math.radians(79.6) / 2)
    seen = sensor[:, 2] > 0.3
    column, row = torch.round(3.0 - focal * sensor[seen, 1] / sensor[seen, 2]).long(), torch.round(3.0 + focal * sensor[seen, 0] / sensor[seen, 2]).long()
    inside = (column >= 0) & (column < 7) & (row >= 0) & (row < 7)
    mask = torch.zeros(7, 7, dtype=torch.bool)
    mask[row[inside], column[inside]] = True
    if not mask.any():
        mask[3:, 2:5] = True
        return mask.flatten()
    return (torch.nn.functional.max_pool2d(mask[None, None].float(), 3, stride=1, padding=1)[0, 0] > 0).flatten()


class Comparator(nn.Module):
    """A transformer over the past frames, the frames to come (their poses only), the text and the image grid."""

    def __init__(self, width: int = 512, layers: int = 8, heads: int = 8, text_dim: int = 768, image_dim: int = 384, cells: int = 49, dropout: float = 0.0,
                 grid: bool = False) -> None:
        super().__init__()
        self.width, self.grid = width, grid
        self.pose_in = nn.Linear(9, width)
        self.image_in = nn.Sequential(nn.LayerNorm(image_dim), nn.Linear(image_dim, width))
        self.time_in = nn.Linear(width, width)
        self.kind = nn.Parameter(torch.randn(4, width) * 0.02)  # past, to come, text, image grid
        self.text_in = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, width))
        self.no_text = nn.Parameter(torch.randn(width) * 0.02)
        self.patch_in = nn.Sequential(nn.LayerNorm(image_dim), nn.Linear(image_dim, width))
        self.patch_where = nn.Parameter(torch.randn(cells, width) * 0.02)
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, 2 * (CELLS * PARTS if grid else image_dim)))

    def forward(self, poses: torch.Tensor, image: torch.Tensor, seen: torch.Tensor, text: torch.Tensor, text_valid: torch.Tensor,
                patches: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """poses (B, 105, 9) standardized, image (B, 30, 384) of the past frames, seen (B, 30), text (B, S, 768), text_valid (B, S),
        patches (B, 49, 384) -> the feature of the 75 frames to come (B, 75, 384) and the log of its variance;
        with ``grid`` the reduced image grid of every fifth of them, (B, 15, 49, 64)."""
        size, device = poses.shape[0], poses.device
        seconds = (torch.arange(HISTORY + FUTURE, device=device) - NOW) / RATE
        tokens = self.pose_in(poses) + self.time_in(time_embedding(seconds[None], self.width))
        before = tokens[:, :HISTORY] + self.image_in(image) + self.kind[0]
        after = tokens[:, HISTORY:] + self.kind[1]
        has_text = text_valid.any(dim=1)
        words = torch.where(has_text[:, None, None], self.text_in(text) + self.kind[2], self.no_text)
        valid = torch.where(has_text[:, None], text_valid, torch.arange(text.shape[1], device=device)[None] == 0)
        cells = self.patch_in(patches) + self.patch_where + self.kind[3]
        sequence = torch.cat([before, after, words.to(before.dtype), cells.to(before.dtype)], dim=1)
        padding = torch.cat([~seen, torch.zeros(size, FUTURE, dtype=torch.bool, device=device), ~valid,
                             torch.zeros(cells.shape[:2], dtype=torch.bool, device=device)], dim=1)
        hidden = self.blocks(sequence, src_key_padding_mask=padding)[:, HISTORY : HISTORY + FUTURE]
        if self.grid:
            mean, spread = self.out(hidden[:, KEPT]).reshape(size, len(KEPT), 2, CELLS, PARTS).unbind(2)
        else:
            mean, spread = self.out(hidden).chunk(2, dim=-1)
        return mean, spread.clamp(-6.0, 3.0)


def predict(model: Comparator, batch: dict, scale: dict, history: torch.Tensor | int = HISTORY, amp: bool = False):
    """The predicted feature (B, 75, 384), its variance, and the observed feature, all standardized."""
    poses = (anchored(batch) - scale["pose_mean"]) / scale["pose_std"]
    size, device = poses.shape[0], poses.device
    history = torch.full((size,), history, device=device) if isinstance(history, int) else history
    seen = torch.arange(HISTORY, device=device)[None] >= HISTORY - history[:, None]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        mean, spread = model(poses, (batch["image"] - scale["image_mean"]) / scale["image_std"], seen, batch["text"], batch["text_valid"], batch["patches"])
    if model.grid:
        return mean.float(), spread.float().exp(), reduced(batch["grid_target"], scale)
    return mean.float(), spread.float().exp(), (batch["target"] - scale["image_mean"]) / scale["image_std"]


def reduced(grid: torch.Tensor, scale: dict) -> torch.Tensor:
    """Image grids (..., 49, 384) -> their 64 principal components, standardized."""
    return ((grid - scale["cell_mean"]) @ scale["cell_basis"]) / scale["cell_std"]


def loss_of(model: Comparator, batch: dict, scale: dict, generator: torch.Generator | None = None, amp: bool = False, random_history: bool = True) -> torch.Tensor:
    """The negative Gaussian log-likelihood of the observed features, per value."""
    size, device = batch["pose"].shape[0], batch["pose"].device
    history = (torch.rand(size, generator=generator, device=device) * HISTORY).long() + 1 if random_history else HISTORY
    mean, variance, target = predict(model, batch, scale, history, amp)
    return 0.5 * (variance.log() + (target - mean) ** 2 / variance).mean()


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    out = np.empty(len(values))
    out[order] = np.arange(len(values))
    return out


@torch.no_grad()
def evaluate(model: Comparator, batches: list, scale: dict, device: str, amp: bool) -> dict:
    model.eval()
    columns = {key: [] for key in ("mismatch", "copy", "surprise", "variance", "turn", "seconds")}
    loss, count = 0.0, 0
    for batch in batches:
        batch = to_device(batch, device)
        mean, variance, target = predict(model, batch, scale, amp=amp)
        error = (target - mean) ** 2
        loss += float(0.5 * (variance.log() + error / variance).mean()) * len(mean)
        count += len(mean)
        turn = frames.heading(anchored(batch)[:, HISTORY:, 3:]).abs() * 180 / math.pi
        seconds = ((torch.arange(FUTURE) + 1) / RATE).expand(len(mean), -1)
        if model.grid:  # (B, 15, 49, 64): average over cells and components
            last = reduced(batch["patches"], scale)[:, None]
            over, turn, seconds = (-2, -1), turn[:, KEPT], seconds[:, KEPT]
        else:
            last, over = (batch["image"][:, -1:] - scale["image_mean"]) / scale["image_std"], -1
        columns["mismatch"].append(error.mean(over).cpu())
        columns["copy"].append(((target - last) ** 2).mean(over).cpu())
        columns["surprise"].append((error / variance).mean(over).cpu())
        columns["variance"].append(variance.mean(over).cpu())
        columns["turn"].append(turn.cpu())
        columns["seconds"].append(seconds)
    a = {key: torch.cat(value).double().numpy().ravel() for key, value in columns.items()}
    result = {"loss": loss / count, "mismatch": float(a["mismatch"].mean()), "copy": float(a["copy"].mean()), "surprise": float(a["surprise"].mean()),
              "rank_correlation": float(np.corrcoef(ranks(a["variance"]), ranks(a["mismatch"]))[0, 1]), "by_turn": [], "by_seconds": []}
    for lo, hi in zip(TURNS[:-1], TURNS[1:]):
        mask = (a["turn"] >= lo) & (a["turn"] < hi)
        result["by_turn"].append({"degrees": [lo, hi], "mismatch": float(a["mismatch"][mask].mean()), "surprise": float(a["surprise"][mask].mean())})
    for lo, hi in ((0, 1), (1, 2), (2, 5)):
        mask = (a["seconds"] > lo) & (a["seconds"] <= hi)
        result["by_seconds"].append({"seconds": [lo, hi], "mismatch": float(a["mismatch"][mask].mean()), "copy": float(a["copy"][mask].mean()), "surprise": float(a["surprise"][mask].mean())})
    model.train()
    return result


def load(folder: Path, device: str) -> tuple[Comparator, dict, dict]:
    """A trained comparator (the state with the lowest held-out loss, if it was kept), the means and deviations it standardizes with, and its settings."""
    saved = torch.load(folder / ("best.pt" if (folder / "best.pt").exists() else "last.pt"), map_location=device)
    config = saved["config"]
    model = Comparator(config["width"], config["layers"], config["heads"], dropout=float(config.get("dropout", 0.0)), grid=config.get("target") == "grid").to(device)
    model.load_state_dict(saved["model"])
    return model.eval(), {key: value.to(device) for key, value in saved["scale"].items()}, config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--moments", type=Path, default=Path("/ptmp/enazaren/gray/moments/moments.npz"))
    parser.add_argument("--tracks", type=Path, default=Path("/ptmp/enazaren/gray/tracks"))
    parser.add_argument("--images", type=Path, default=Path("/ptmp/enazaren/gray/pinhole"))
    parser.add_argument("--patches", type=Path, default=Path("/ptmp/enazaren/gray/grids"))
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--text-drop", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--target", choices=("token", "grid"), default="token", help="what is predicted: the class token of every frame, or the reduced 7 x 7 grid of every fifth")
    parser.add_argument("--everything", action="store_true", help="train on all moments of the training locations, not only the navigation subset")
    parser.add_argument("--wide", type=int, default=None, help="move now by up to this many frames either way in training (37 covers a whole sentence)")
    parser.add_argument("--revisits", type=Path, default=None, help="folder of gray/revisits.py: train on revisits of the frames as targets, so that the uncertainty covers how an unchanged place varies")
    parser.add_argument("--revisit-share", type=float, default=0.5, help="how often a target frame is replaced by its revisit, where one exists")
    parser.add_argument("--eval-every", type=int, default=2500)
    parser.add_argument("--val-moments", type=int, default=1024)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.version.hip:
        torch.backends.cuda.preferred_blas_library("cublas")
    hard = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = args.amp and device == "cuda"

    data = dict(listing=args.moments, tracks=args.tracks, images=args.images, cache=args.cache, patches=args.patches, future=True, future_grid=args.target == "grid")
    training = Moments(**data, held=False, jitter=True, everything=args.everything, wide=args.wide, revisits=args.revisits, revisit_share=args.revisit_share)
    held = Moments(**data, held=True)  # the held-out moments keep their true frames
    print(f"{len(training):,d} training moments, {len(held):,d} held-out moments, device {device}", flush=True)
    options = dict(batch_size=args.batch_size, num_workers=args.workers, persistent_workers=args.workers > 0, collate_fn=collate)
    val_batches = list(DataLoader(Subset(held, np.random.default_rng(args.seed).permutation(len(held))[: args.val_moments].tolist()), **options))

    last = args.save_dir / "last.pt" if args.save_dir else None
    saved = torch.load(last, map_location=device) if args.resume and last and last.is_file() else None
    if saved:
        scale = {key: value.to(device) for key, value in saved["scale"].items()}
    else:
        some = Subset(training, np.random.default_rng(args.seed).permutation(len(training))[:2048].tolist())
        batches = list(DataLoader(some, batch_size=256, collate_fn=collate, num_workers=args.workers))
        poses, images = torch.cat([anchored(b) for b in batches]), torch.cat([torch.cat([b["image"], b["target"]], dim=1) for b in batches])
        scale = {"pose_mean": poses.mean((0, 1)), "pose_std": poses.std((0, 1)).clamp_min(1e-2), "image_mean": images.mean((0, 1)), "image_std": images.std((0, 1)).clamp_min(1e-4)}
        if args.target == "grid":  # the principal components of the grid cells, from the grids of the frames at now
            cells = torch.cat([b["patches"] for b in batches]).flatten(0, 1)
            centre = cells.mean(0)
            _u, size, basis = torch.pca_lowrank(cells - centre, q=PARTS, center=False, niter=4)
            scale |= {"cell_mean": centre, "cell_basis": basis, "cell_std": ((cells - centre) @ basis).std(0).clamp_min(1e-4)}
            print(f"grid cells: {PARTS} components keep {float((size**2).sum() / ((cells - centre) ** 2).sum()):.0%} of the variance", flush=True)
        scale = {key: value.to(device) for key, value in scale.items()}

    model = Comparator(args.width, args.layers, args.heads, dropout=args.dropout, grid=args.target == "grid").to(device)
    best = min((json.loads(line)["loss"] for line in (args.save_dir / "log.jsonl").read_text().splitlines()), default=float("inf")) if saved and args.save_dir else float("inf")
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters", flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    step = 0
    if saved:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        schedule.load_state_dict(saved["schedule"])
        step = saved["step"]
        print(f"resumed at step {step}", flush=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        (args.save_dir / "config.json").write_text(json.dumps(config, indent=1))

    loader = DataLoader(training, shuffle=True, drop_last=True, **options)
    running, started = [], time.time()
    model.train()
    while step < args.steps:
        for batch in loader:
            batch = to_device(batch, device)
            batch["text_valid"] = batch["text_valid"] & (torch.rand(len(batch["pose"]), device=device) >= args.text_drop)[:, None]
            loss = loss_of(model, batch, scale, amp=amp)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            step += 1
            running.append(loss.item())
            if step % args.eval_every == 0 or step == args.steps:
                train_loss, running = float(np.mean(running)), []
                r = evaluate(model, val_batches, scale, device, amp)
                print(f"step {step} | loss {train_loss:.4f} | held-out loss {r['loss']:.4f} | mismatch {r['mismatch']:.3f} (copying the last frame {r['copy']:.3f}) | "
                      f"surprise {r['surprise']:.2f} | uncertainty against error, rank correlation {r['rank_correlation']:.2f} | {time.time() - started:.0f} s", flush=True)
                print("    by turn since now: " + "; ".join(f"{t['degrees'][0]} to {t['degrees'][1]} deg mismatch {t['mismatch']:.3f}, surprise {t['surprise']:.2f}" for t in r["by_turn"]), flush=True)
                print("    by time: " + "; ".join(f"{t['seconds'][0]} to {t['seconds'][1]} s mismatch {t['mismatch']:.3f} (copy {t['copy']:.3f}), surprise {t['surprise']:.2f}" for t in r["by_seconds"]), flush=True)
                if args.save_dir:
                    with (args.save_dir / "log.jsonl").open("a") as handle:
                        handle.write(json.dumps({"step": step, "train_loss": train_loss, **r}) + "\n")
                    state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "schedule": schedule.state_dict(), "step": step, "config": config,
                             "scale": {key: value.cpu() for key, value in scale.items()}}
                    torch.save(state, last)
                    if r["loss"] < best:  # the state that is best on held-out moments is the one to use: later ones fit the training rooms
                        best = r["loss"]
                        torch.save(state, args.save_dir / "best.pt")
            if step >= args.steps:
                break


if __name__ == "__main__":
    main()
