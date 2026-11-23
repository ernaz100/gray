"""Run gray/joint.py end to end on a tiny made-up cache (CPU, about a minute).

    python tests/smoke_joint.py

The made-up world: image features are linear in how far the view has turned, the pelvis is 10 cm
behind the camera, the body faces where the camera looks and the pose latents follow the turning
speed. So along the real head path the expected feature must beat copying, and the motion
generated along the real head path must put the pelvis far closer than "stay where you are".
The run is stopped half way and resumed, to test that too.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tests.smoke_features import make_cache  # noqa: E402


def add_motion_and_text(cache: Path) -> None:
    rng = np.random.default_rng(1)
    index = json.loads((cache / "index.json").read_text())
    for meta in index["sequences"]:
        folder = cache / meta["sequence_id"]
        head = np.load(folder / "head.npy")
        yaw = -np.arctan2(head[..., 5], head[..., 3])  # the cache's first row is (cos(-yaw), 0, sin(-yaw))
        root = np.stack([head[..., 0] + 0.1 * np.sin(yaw), head[..., 2] - 0.1 * np.cos(yaw), np.sin(yaw), np.cos(yaw)], axis=-1)
        np.save(folder / "root.npy", root.astype(np.float32))
        speed = np.gradient(yaw, axis=1)[..., None] * 20
        np.save(folder / "latents.npy", (speed * np.linspace(-1, 1, 32) + rng.normal(size=(*yaw.shape, 32)) * 0.05).astype(np.float32))
        np.save(folder / "text_tokens.npy", rng.normal(size=(5, 77, 768)).astype(np.float16))
        np.save(folder / "text_lengths.npy", rng.integers(4, 12, size=5).astype(np.int16))
        text_index = rng.integers(-1, 5, size=(head.shape[0], 4)).astype(np.int16)
        text_index[::3] = -1  # every third window has no text
        np.save(folder / "text_index.npy", text_index)
    index.update(latent_mean=0.0, latent_sigma=1.0)
    (cache / "index.json").write_text(json.dumps(index))


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as folder:
        cache, out = Path(folder) / "cache", Path(folder) / "out"
        make_cache(cache)
        add_motion_and_text(cache)
        for steps in ("300", "600"):
            subprocess.run([
                sys.executable, "-m", "gray.joint", "--cache", str(cache), "--save-dir", str(out), "--resume",
                "--steps", steps, "--eval-every", "300", "--batch-size", "32", "--width", "96", "--layers", "2", "--heads", "4",
                "--workers", "0", "--val-windows", "64", "--warmup", "50", "--holdout", "0.34", "--lr", "1e-3", "--sample-steps", "8",
            ], cwd=ROOT, check=True)
        log = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
        assert [entry["step"] for entry in log] == [300, 600]
        for start in ("warm", "cold"):
            final = log[-1][start]
            print(start, {k: round(final[k][-1], 3) for k in ("feature_mse", "feature_copy_mse", "motion_pelvis_cm", "head_stay_cm", "head_cm")})
            assert final["feature_mse"][-1] < final["feature_copy_mse"][-1]
            assert final["motion_pelvis_cm"][-1] < 0.5 * final["head_stay_cm"][-1]
        print("ok")
