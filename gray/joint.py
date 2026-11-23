#!/usr/bin/env python3
"""The joint model: head path, image features and body motion in one token per frame.

    python -m gray.joint --save-dir <dir> --amp --resume

One token per 15 Hz frame of an 8 s window, in the frame anchored at "now" (``frames.anchor``):

    head      9    camera position and 6D rotation
    feature   384  DINOv3 class token of the frame
    motion    36   MoLingo pose latent (32) and STEER's root channels (4)

All values are standardized. Every part of every token has its own noise level in [0, 1]:
1 is clean, 0 is pure noise, ``x = (1 - level) noise + level value`` (flow matching, as STEER's
train.py). The model predicts the clean value, not the velocity ``value - noise``: the feature
part alone has 384 values, and a velocity output would have to carry all of that noise through
the network. Text is CLIP tokens next to the frame tokens.

Training picks "now" among the first ``--max-history`` tokens. Frames up to now are clean, as
they were observed. For a share of the clips (``--cold``) the body motion up to now is not
known, as at the very start of a deployment. The parts of the later frames are noised by stage:

    head       head noisy, feature and motion pure noise        image, text -> head path
    feature    head clean, feature noisy, motion pure noise     head path -> what the camera shows
    motion     head and feature clean, motion noisy             both -> body motion
    seen-body  head and motion clean, feature noisy             does knowing the body help the features?
    free       three independent levels

Sampling runs the first three stages in this order (``generate``). The comparator uses the
feature stage with the head poses that were really reached; one step from pure noise gives the
expected feature (``expected``), which is what experiment 1 regressed.

Every ``--eval-every`` steps, on held-out locations, for a warm start (1 s of history) and a cold
start (one frame, no body):

    chain            head path, features and motion generated in order; head and pelvis error
    feature          expected feature along the real head path, against copying the frame at now,
                     and the same when the real body motion is given too
    motion           motion generated along the real head path and features
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
from torch.utils.data import DataLoader, Dataset

from gray import frames
from gray.features import CACHE, RATE, TOKENS, split, time_embedding

HEAD, FEATURE, MOTION = 0, 1, 2
PARTS = ("head", "feature", "motion")
SIZES = (9, 384, 36)
LATENT, ROOT = 32, 4
DIM = sum(SIZES)
PART_OF = torch.repeat_interleave(torch.arange(3), torch.tensor(SIZES))  # the part of each of the 429 values
MAX_TEXT = 128  # tokens kept when several sentences overlap a window
NAN = float("nan")
STAGES = {  # level of (head, feature, motion) in the frames after now; nan is drawn uniformly
    "head": (NAN, 0.0, 0.0),
    "feature": (1.0, NAN, 0.0),
    "motion": (1.0, 1.0, NAN),
    "seen-body": (1.0, NAN, 1.0),
    "free": (NAN, NAN, NAN),
}
SECONDS = (1.0, 2.0, 4.0)  # edges of the horizons errors are reported for


class Windows(Dataset):
    """Cached windows with everything the joint model needs; arrays are memory-mapped and opened on first use."""

    NAMES = ("dino", "starts", "head", "latents", "root", "text_index", "text_tokens", "text_lengths")

    def __init__(self, cache: Path, items: list[tuple[str, int]], images: Path | None = None, patches: Path | None = None,
                 now: int | str = "random", max_history: int = 30) -> None:
        """``images``: a folder with <sequence>/dino.npy to use in place of the cache's image features (gray/pinhole.py).

        ``patches``: a folder of gray/patches.py. The window then comes with a "now" and the image grid of that frame.
        ``now`` is a fixed token, "random" (training) or "spread" (different per window, the same every time).
        """
        self.cache, self.items, self.images, self.patches, self.now, self.max_history = cache, items, images, patches, now, max_history
        self._grids: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._open: dict[str, dict[str, np.ndarray]] = {}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sequence, window = self.items[index]
        if sequence not in self._open:
            self._open[sequence] = {n: np.load(self.cache / sequence / f"{n}.npy", mmap_mode="r") for n in self.NAMES}
            if self.images is not None:
                other = np.load(self.images / sequence / "dino.npy", mmap_mode="r")
                if other.shape != self._open[sequence]["dino"].shape:
                    raise ValueError(f"{sequence}: {other.shape} image features in {self.images}, {self._open[sequence]['dino'].shape} in the cache")
                self._open[sequence]["dino"] = other
        a = self._open[sequence]
        first = int(a["starts"][window]) // 2
        sentences = [int(i) for i in a["text_index"][window] if i >= 0]
        if sentences:
            text = np.concatenate([a["text_tokens"][i, : a["text_lengths"][i]] for i in sentences])[:MAX_TEXT]
        else:
            text = np.zeros((0, a["text_tokens"].shape[-1]))
        tensor = lambda values: torch.from_numpy(np.array(values, dtype=np.float32))
        out = {"head": tensor(a["head"][window]), "image": tensor(a["dino"][first : first + TOKENS]),
               "latents": tensor(a["latents"][window]), "root": tensor(a["root"][window]), "text": tensor(text)}
        if self.patches is not None:
            if sequence not in self._grids:
                folder = self.patches / sequence
                self._grids[sequence] = (np.load(folder / "rows.npy"), np.load(folder / "patches.npy", mmap_mode="r"))
            rows, grids = self._grids[sequence]
            now = {"random": int(torch.randint(0, self.max_history, (1,))), "spread": (window * 7919 + index) % self.max_history}.get(self.now, self.now)
            row = int(rows[first + now])
            if row < 0:
                raise KeyError(f"{sequence}, window {window}, frame {now}: no image grid in {self.patches}")
            out["now"], out["patches"] = torch.tensor(now), tensor(grids[row])
        return out


def restrict(items: list[tuple[str, int]], subset: Path, going: bool = False) -> list[tuple[str, int]]:
    """The windows that are also in a subset file of gray/census.py (navigation.npz); ``going``: only those that go somewhere."""
    listed = np.load(subset)
    keep = listed["going"] if going else np.ones(len(listed["window"]), dtype=bool)
    wanted = set(zip(listed["sequence"][keep].tolist(), listed["window"][keep].tolist()))
    return [item for item in items if item in wanted]


def collate(rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    batch = {key: torch.stack([row[key] for row in rows]) for key in ("head", "image", "latents", "root", "now", "patches") if key in rows[0]}
    longest = max(1, max(row["text"].shape[0] for row in rows))
    batch["text"] = torch.zeros(len(rows), longest, rows[0]["text"].shape[-1])
    batch["text_valid"] = torch.zeros(len(rows), longest, dtype=torch.bool)
    for i, row in enumerate(rows):
        count = row["text"].shape[0]
        batch["text"][i, :count] = row["text"]
        batch["text_valid"][i, :count] = True
    return batch


def raw_tokens(batch: dict[str, torch.Tensor], now: torch.Tensor) -> torch.Tensor:
    """(B, L, 429) tokens in the frame anchored at ``now``, not standardized."""
    return torch.cat([frames.anchor(batch["head"], now), batch["image"], batch["latents"],
                      frames.anchor_root(batch["root"], batch["head"], now)], dim=-1)


def seconds_from(now: torch.Tensor, length: int = TOKENS) -> torch.Tensor:
    return (torch.arange(length, device=now.device)[None] - now[:, None]) / RATE


def level_embedding(level: torch.Tensor, dim: int = 256) -> torch.Tensor:
    frequency = torch.exp(-math.log(10000.0) * torch.arange(dim // 2, device=level.device) / (dim // 2))
    angle = level[..., None].float() * 1000.0 * frequency
    return torch.cat([angle.cos(), angle.sin()], dim=-1)


class Joint(nn.Module):
    """A transformer over the frame tokens and the text tokens; predicts the clean value of all three parts."""

    def __init__(self, width: int = 512, layers: int = 8, heads: int = 8, text_dim: int = 768, patches: bool = False, patch_dim: int = 384,
                 cells: int = 49) -> None:
        """``patches``: the model also attends to a grid of image features of the frame at now (gray/patches.py), set with ``see``."""
        super().__init__()
        self.width = width
        self.seen: torch.Tensor | None = None
        if patches:
            self.patch_in = nn.Sequential(nn.LayerNorm(patch_dim), nn.Linear(patch_dim, width))
            self.patch_where = nn.Parameter(torch.randn(cells, width) * 0.02)
        self.token_in = nn.Linear(DIM, width)
        self.time_in = nn.Linear(width, width)
        self.level_in = nn.ModuleList([nn.Sequential(nn.Linear(256, width), nn.SiLU(), nn.Linear(width, width)) for _ in PARTS])
        self.text_in = nn.Sequential(nn.LayerNorm(text_dim), nn.Linear(text_dim, width))
        self.text_kind = nn.Parameter(torch.zeros(width))
        self.no_text = nn.Parameter(torch.randn(width) * 0.02)
        layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, DIM))

    def see(self, patches: torch.Tensor | None) -> None:
        """The image grid (B, 49, 384) of the frame at now for the next calls; it is not an argument, so sampling code need not carry it."""
        self.seen = patches

    def forward(self, x: torch.Tensor, levels: torch.Tensor, seconds: torch.Tensor, text: torch.Tensor, text_valid: torch.Tensor) -> torch.Tensor:
        """x (B, L, 429) noised tokens, levels (B, L, 3), seconds (B, L) from now, text (B, S, 768), text_valid (B, S)."""
        tokens = self.token_in(x) + self.time_in(time_embedding(seconds, self.width)).to(x.dtype)
        for part, embed in enumerate(self.level_in):
            tokens = tokens + embed(level_embedding(levels[..., part]).to(x.dtype))
        has_text = text_valid.any(dim=1)
        words = self.text_in(text) + self.text_kind
        words = torch.where(has_text[:, None, None], words, self.no_text.to(words.dtype))
        # A clip without text keeps one placeholder token, so the text side is never empty.
        valid = torch.where(has_text[:, None], text_valid, torch.arange(text.shape[1], device=x.device)[None] == 0)
        padding = torch.cat([torch.zeros_like(seconds, dtype=torch.bool), ~valid], dim=1)
        sequence = [tokens, words.to(tokens.dtype)]
        if hasattr(self, "patch_in"):
            if self.seen is None or self.seen.shape[0] != x.shape[0]:
                raise ValueError("this model needs the image grid of the frame at now: call see(patches) with the batch first")
            cells = self.patch_in(self.seen.float()) + self.patch_where
            sequence.append(cells.to(tokens.dtype))
            padding = torch.cat([padding, torch.zeros(cells.shape[:2], dtype=torch.bool, device=x.device)], dim=1)
        hidden = self.blocks(torch.cat(sequence, dim=1), src_key_padding_mask=padding)
        return self.out(hidden[:, : x.shape[1]])


def training_plan(size: int, max_history: int, cold: float, share: torch.Tensor, device: str,
                  generator: torch.Generator | None = None, now: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """"Now" (B,) and the noise levels (B, L, 3) of a training batch; level 1 marks what is given. ``now``: use these instead of drawing them."""
    draw = lambda *shape: torch.rand(*shape, generator=generator, device=device)
    drawn = (draw(size) * max_history).long().clamp_max(max_history - 1)
    now = drawn if now is None else now
    past = torch.arange(TOKENS, device=device)[None] <= now[:, None]
    stage = torch.multinomial(share.to(device), size, replacement=True, generator=generator)
    fixed = torch.tensor(list(STAGES.values()), device=device)[stage]  # (B, 3)
    level = torch.where(fixed.isnan(), draw(size, 3), fixed)
    known = past[..., None].repeat(1, 1, 3)
    known[..., MOTION] &= (draw(size) >= cold)[:, None]
    return now, torch.where(known, torch.ones((), device=device), level[:, None, :].expand(-1, TOKENS, -1))


def flow_loss(model: Joint, x1: torch.Tensor, levels: torch.Tensor, seconds: torch.Tensor, text: torch.Tensor, text_valid: torch.Tensor,
              root_weight: float, generator: torch.Generator | None = None, amp: bool = False,
              weights: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> tuple[torch.Tensor, dict[str, float]]:
    """Mean squared error of the predicted clean values, per part over what is not given; the loss is their sum, weighted by ``weights``."""
    noise = torch.randn(x1.shape, generator=generator, device=x1.device)
    level = levels[..., PART_OF.to(x1.device)]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        clean = model((1 - level) * noise + level * x1, levels, seconds, text, text_valid)
    error = (clean.float() - x1) ** 2
    head, feature, motion = error.split(SIZES, dim=-1)
    per_frame = [head.mean(-1), feature.mean(-1),
                 (motion[..., :LATENT].mean(-1) + root_weight * motion[..., LATENT:].mean(-1)) / (1 + root_weight)]
    parts = [(e * (levels[..., p] < 1)).sum() / (levels[..., p] < 1).sum().clamp_min(1) for p, e in enumerate(per_frame)]
    return sum(w * part for w, part in zip(weights, parts)), {name: value.item() for name, value in zip(PARTS, parts)}


def _start(x: torch.Tensor, known: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    noise = torch.randn(x.shape, generator=generator, device=x.device)
    return torch.where(known[..., PART_OF.to(x.device)], x, noise)


@torch.no_grad()
def generate(model: Joint, x: torch.Tensor, known: torch.Tensor, seconds: torch.Tensor, text: torch.Tensor, text_valid: torch.Tensor,
             order: tuple[int, ...] = (HEAD, FEATURE, MOTION), steps: int = 20, generator: torch.Generator | None = None,
             amp: bool = False) -> torch.Tensor:
    """Fill in what is not known, one part after the other, with Euler steps.

    x (B, L, 429) holds the true values where ``known`` (B, L, 3) is set. While one part is denoised,
    the parts before it in ``order`` are finished and the ones after it stay pure noise. Parts that
    are neither known nor in ``order`` come back as noise.
    """
    part_of = PART_OF.to(x.device)
    x, levels = _start(x, known, generator), known.float()
    for part in order:
        free = ~known[..., part]
        for step in range(steps):
            levels[..., part] = torch.where(free, step / steps, 1.0)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                clean = model(x, levels, seconds, text, text_valid).float()
            velocity = (clean - x) / (1 - step / steps)  # the last step lands on the prediction
            x = x + velocity * (free[..., None] & (part_of == part)) / steps
        levels[..., part] = 1.0
    return x


@torch.no_grad()
def expected(model: Joint, x: torch.Tensor, known: torch.Tensor, seconds: torch.Tensor, text: torch.Tensor, text_valid: torch.Tensor,
             generator: torch.Generator | None = None, amp: bool = False) -> torch.Tensor:
    """The expected value of everything not known, in one step from pure noise (known values are returned as they are)."""
    x = _start(x, known, generator)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        clean = model(x, known.float(), seconds, text, text_valid).float()
    return torch.where(known[..., PART_OF.to(x.device)], x, clean)


def to_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


@torch.no_grad()
def evaluate(model: Joint, batches: list, mean: torch.Tensor, std: torch.Tensor, device: str, history: int, cold: bool,
             steps: int, amp: bool) -> dict:
    """Errors on the frames after now by horizon, for a fixed history. See the module text for the three tests."""
    model.eval()
    generator = torch.Generator(device=device).manual_seed(0)
    sums: dict[str, torch.Tensor] = {}
    for batch in batches:
        batch = to_device(batch, device)
        size = batch["head"].shape[0]
        model.see(batch.get("patches"))
        now = torch.full((size,), history - 1, device=device)
        seconds = seconds_from(now)
        raw = raw_tokens(batch, now)
        x1 = (raw - mean) / std
        text = (batch["text"], batch["text_valid"])
        known = (seconds <= 0)[..., None].repeat(1, 1, 3)
        if cold:
            known[..., MOTION] = False
        given = lambda *parts: torch.stack([known[..., p] | (p in parts) for p in range(3)], dim=-1)
        run = dict(generator=generator, amp=amp)
        chain = generate(model, x1, known, seconds, *text, steps=steps, **run) * std + mean
        feature = expected(model, x1, given(HEAD), seconds, *text, **run)
        feature_body = expected(model, x1, given(HEAD, MOTION), seconds, *text, **run)
        motion = generate(model, x1, given(HEAD, FEATURE), seconds, *text, order=(MOTION,), steps=steps, **run) * std + mean

        f = slice(SIZES[0], SIZES[0] + SIZES[1])
        root = lambda tokens: tokens[..., -ROOT:]
        yaw = lambda tokens: torch.atan2(root(tokens)[..., 2], root(tokens)[..., 3])
        turn = lambda angle: torch.atan2(angle.sin(), angle.cos()).abs() * 180 / math.pi
        at_now = raw[torch.arange(size, device=device), now][:, None]
        values = {
            "head_cm": (chain[..., :3] - raw[..., :3]).norm(dim=-1) * 100,
            "head_stay_cm": (at_now[..., :3] - raw[..., :3]).norm(dim=-1) * 100,
            "heading_deg": turn(frames.heading(chain[..., 3:9]) - frames.heading(raw[..., 3:9])),
            "heading_stay_deg": turn(frames.heading(raw[..., 3:9])),
            "pelvis_cm": (root(chain)[..., :2] - root(raw)[..., :2]).norm(dim=-1) * 100,
            "feature_mse": ((feature[..., f] - x1[..., f]) ** 2).mean(-1),
            "feature_body_mse": ((feature_body[..., f] - x1[..., f]) ** 2).mean(-1),
            "feature_copy_mse": ((x1[torch.arange(size, device=device), now][:, None, f] - x1[..., f]) ** 2).mean(-1),
            "motion_pelvis_cm": (root(motion)[..., :2] - root(raw)[..., :2]).norm(dim=-1) * 100,
            "motion_yaw_deg": turn(yaw(motion) - yaw(raw)),
            "motion_latent_mse": (((motion - raw) / std)[..., -SIZES[2] : -ROOT] ** 2).mean(-1),
            "frames": torch.ones_like(seconds),
        }
        future = seconds > 0
        index = torch.bucketize(seconds[future], torch.tensor(SECONDS, device=device))
        for name, value in values.items():
            sums.setdefault(name, torch.zeros(len(SECONDS) + 1, dtype=torch.float64, device=device)).index_add_(0, index, value[future].double())
    model.train()
    edges = (0.0, *SECONDS, TOKENS / RATE)
    result = {"horizons": [f"{a:g} to {b:g} s" for a, b in zip(edges[:-1], edges[1:])]}
    for name, total in sums.items():
        if name != "frames":
            result[name] = [float(v) for v in (total / sums["frames"]).cpu()] + [float(total.sum() / sums["frames"].sum())]
    return result


def report(result: dict) -> str:
    names = [n for n in result if n != "horizons"]
    lines = [f"  {'':<20}" + "".join(f"{h:>12}" for h in (*result["horizons"], "all"))]
    lines += [f"  {name:<20}" + "".join(f"{v:12.3f}" for v in result[name]) for name in names]
    return "\n".join(lines)


def load(folder: Path, device: str, name: str = "last.pt") -> tuple[Joint, torch.Tensor, torch.Tensor, dict]:
    """A trained model from its run folder, with the mean and standard deviation of its tokens."""
    saved = torch.load(folder / name, map_location=device)
    config = saved["config"]
    model = Joint(config["width"], config["layers"], config["heads"], patches=bool(config.get("patches"))).to(device)
    model.load_state_dict(saved["model"])
    return model.eval(), saved["mean"].to(device), saved["std"].to(device), config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--subset", type=Path, default=None, help="train and evaluate only on the windows listed in this file (navigation.npz of gray/census.py)")
    parser.add_argument("--part-weights", type=float, nargs=3, default=[1.0, 1.0, 1.0], metavar=("HEAD", "FEATURE", "MOTION"),
                        help="weights of the three parts in the loss; 1 0 0 trains the head path only")
    parser.add_argument("--patches", type=Path, default=None, help="folder of gray/patches.py: the model also sees a grid of image features of the frame at now")
    parser.add_argument("--images", type=Path, default=None, help="folder with <sequence>/dino.npy replacing the cache's image features")
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--resume", action="store_true", help="continue from last.pt in --save-dir if it is there")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--max-history", type=int, default=30, help="now is one of the first this many tokens (15 per second)")
    parser.add_argument("--cold", type=float, default=0.25, help="share of clips whose body motion up to now is not given")
    parser.add_argument("--stages", type=float, nargs=5, default=[0.25, 0.25, 0.25, 0.1, 0.15], metavar="SHARE",
                        help="shares of the stages " + ", ".join(STAGES))
    parser.add_argument("--text-drop", type=float, default=0.1, help="share of clips trained without their text")
    parser.add_argument("--root-weight", type=float, default=1.0, help="weight of the four root channels against the 32 pose latents")
    parser.add_argument("--holdout", type=float, default=0.15)
    parser.add_argument("--min-clock", type=float, default=0.5)
    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--val-windows", type=int, default=1024)
    parser.add_argument("--sample-steps", type=int, default=20)
    parser.add_argument("--amp", action="store_true", help="bf16 mixed precision")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if torch.version.hip:  # Viper's MI300A: rocBLAS, see STEER's paths.setup_torch
        torch.backends.cuda.preferred_blas_library("cublas")
    hard = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))  # eight memory maps per sequence and worker
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = args.amp and device == "cuda"
    torch.manual_seed(args.seed)
    order = np.random.default_rng(args.seed)
    share = torch.tensor(args.stages)

    train_items, val_items = split(args.cache, args.holdout, args.min_clock)
    if args.subset is not None:
        train_items, val_items = restrict(train_items, args.subset), restrict(val_items, args.subset)
    print(f"{len(train_items):,d} training windows, {len(val_items):,d} held-out windows, device {device}", flush=True)
    val_items = [val_items[i] for i in order.permutation(len(val_items))[: args.val_windows]]
    options = dict(batch_size=args.batch_size, num_workers=args.workers, persistent_workers=args.workers > 0, collate_fn=collate)
    val_batches = list(DataLoader(Windows(args.cache, val_items, args.images, args.patches, "spread", args.max_history), **options))
    # with image grids a batch is tied to one "now": the sampling test then runs from 15 frames of history only
    eval_batches = list(DataLoader(Windows(args.cache, val_items, args.images, args.patches, 14), **options)) if args.patches else val_batches

    last = args.save_dir / "last.pt" if args.save_dir else None
    saved = torch.load(last, map_location=device) if args.resume and last and last.is_file() else None
    if saved:
        mean, std = saved["mean"].to(device), saved["std"].to(device)
    else:  # mean and spread of every value, from a sample of training windows anchored at a random now
        index = json.loads((args.cache / "index.json").read_text())
        sample = Windows(args.cache, [train_items[i] for i in order.permutation(len(train_items))[:2048]], args.images)
        tokens = torch.cat([raw_tokens(b, torch.randint(0, args.max_history, (b["head"].shape[0],))) for b in DataLoader(sample, **options)])
        mean, std = tokens.mean((0, 1)), tokens.std((0, 1)).clamp_min(1e-2)
        latent = slice(SIZES[0] + SIZES[1], DIM - ROOT)  # one mean and sigma for all pose latents, as STEER
        mean[latent], std[latent] = index["latent_mean"], index["latent_sigma"]
        mean, std = mean.to(device), std.to(device)

    model = Joint(args.width, args.layers, args.heads, patches=args.patches is not None).to(device)
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
    loader = DataLoader(Windows(args.cache, train_items, args.images, args.patches, "random", args.max_history), shuffle=True, drop_last=True, **options)

    def prepared(batch: dict, generator: torch.Generator | None = None, drop: float = 0.0):
        batch = to_device(batch, device)
        size = batch["head"].shape[0]
        now, levels = training_plan(size, args.max_history, args.cold, share, device, generator, batch.get("now"))
        model.see(batch.get("patches"))
        keep = torch.rand(size, generator=generator, device=device) >= drop
        return (raw_tokens(batch, now) - mean) / std, levels, seconds_from(now), batch["text"], batch["text_valid"] & keep[:, None]

    @torch.no_grad()
    def validation() -> dict[str, float]:
        """The training loss on held-out windows, with the same noise, stages and now every call."""
        model.eval()
        generator = torch.Generator(device=device).manual_seed(0)
        totals = np.mean([list(flow_loss(model, *prepared(b, generator), args.root_weight, generator, amp, tuple(args.part_weights))[1].values()) for b in val_batches], axis=0)
        model.train()
        return dict(zip(PARTS, (float(v) for v in totals)))

    running, started = [], time.time()
    while step < args.steps:
        for batch in loader:
            loss, parts = flow_loss(model, *prepared(batch, drop=args.text_drop), args.root_weight, amp=amp, weights=tuple(args.part_weights))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            step += 1
            running.append(list(parts.values()))
            if step % args.eval_every == 0 or step == args.steps:
                train_loss, running = dict(zip(PARTS, (float(v) for v in np.mean(running, axis=0)))), []
                val = validation()
                scores = {"warm": evaluate(model, eval_batches, mean, std, device, 15, args.cold >= 1, args.sample_steps, amp)}
                if not args.patches:
                    scores["cold"] = evaluate(model, val_batches, mean, std, device, 1, True, args.sample_steps, amp)
                show = lambda losses: " ".join(f"{n} {v:.3f}" for n, v in losses.items())
                print(f"\nstep {step} | loss: {show(train_loss)} | held-out: {show(val)} | {time.time() - started:.0f} s", flush=True)
                for name, result in scores.items():
                    print(f" {name} start\n{report(result)}", flush=True)
                if args.save_dir:
                    with (args.save_dir / "log.jsonl").open("a") as handle:
                        handle.write(json.dumps({"step": step, "train_loss": train_loss, "val_loss": val, **scores}) + "\n")
                    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "schedule": schedule.state_dict(),
                                "step": step, "config": config, "mean": mean.cpu(), "std": std.cpu()}, last)
            if step >= args.steps:
                break


if __name__ == "__main__":
    main()
