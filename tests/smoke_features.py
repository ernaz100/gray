"""Run gray/features.py end to end on a tiny made-up cache (CPU, a few seconds).

    python tests/smoke_features.py

The made-up features are a linear function of how far the view has turned, so the pose-aware
model should beat both copying and the model without poses.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def make_cache(root: Path, sequences: int = 9, windows: int = 30) -> None:
    rng = np.random.default_rng(0)
    mix = rng.normal(size=(1, 384)).astype(np.float32)
    index = []
    for s in range(sequences):
        folder = root / f"seq{s}"
        folder.mkdir(parents=True)
        frames = windows * 30 + 120
        yaw = np.cumsum(rng.normal(size=frames) * 0.05)
        xz = np.cumsum(rng.normal(size=(frames, 2)) * 0.03, axis=0)
        code = yaw[:, None].astype(np.float32)
        np.save(folder / "dino.npy", (code @ mix + rng.normal(size=(frames, 384)) * 0.05).astype(np.float16))
        np.save(folder / "starts.npy", np.arange(windows) * 60)
        head = np.zeros((windows, 120, 9), dtype=np.float32)
        for w in range(windows):
            rows = slice(w * 30, w * 30 + 120)
            c, s_ = np.cos(-yaw[rows]), np.sin(-yaw[rows])
            head[w, :, 0], head[w, :, 1], head[w, :, 2] = xz[rows, 0], 1.6, xz[rows, 1]
            head[w, :, 3], head[w, :, 5], head[w, :, 7] = c, s_, 1.0  # first two rows of a yaw rotation
        np.save(folder / "head.npy", head)
        index.append({"sequence_id": f"seq{s}", "location": f"Loc_{s % 3}", "windows": windows, "clock_gyro_correlation": 0.9})
    (root / "index.json").write_text(json.dumps({"sequences": index}))


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as folder:
        cache, out = Path(folder) / "cache", Path(folder) / "out"
        make_cache(cache)
        for pose in ("full", "none"):
            subprocess.run([
                sys.executable, "-m", "gray.features", "--cache", str(cache), "--save-dir", str(out / pose), "--pose", pose,
                "--steps", "400", "--eval-every", "200", "--batch-size", "32", "--width", "64", "--layers", "2", "--heads", "4",
                "--workers", "0", "--val-windows", "64", "--warmup", "50", "--holdout", "0.34", "--lr", "1e-3",
            ], cwd=ROOT, check=True)
        final = {p: json.loads((out / p / "results.json").read_text())["log"][-1]["history"]["1"]["all"] for p in ("full", "none")}
        print({p: round(r["mse"], 3) for p, r in final.items()}, "copy", round(final["full"]["mse_copy"], 3))
        assert final["full"]["mse"] < final["none"]["mse"] and final["full"]["mse"] < final["full"]["mse_copy"]
        print("ok")
