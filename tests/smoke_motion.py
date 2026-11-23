"""Run gray/motion.py end to end on tiny made-up data (CPU, about a minute).

    python tests/smoke_motion.py

The made-up world is that of tests/smoke_head.py (a head that walks straight ahead after a "go"
sentence and stands after a "stay" sentence), with a body: the pelvis is 0.3 m behind the camera
and faces the way the camera looks, and the first number of the pose latent swings while the
head moves and is zero while it stands. A model that uses the given head path must place the
pelvis far closer than "straight under the camera" (30 cm off), with and without the past body.
The run is stopped half way and resumed. One recording has no body for part of its time.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from smoke_head import make  # noqa: E402

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as folder:
        paths, out = make(Path(folder)), Path(folder) / "out"
        (Path(folder) / "body").mkdir()
        for i, path in enumerate(sorted(paths["tracks"].glob("*.npz"))):
            pose = np.load(path)["pose"]
            rows = np.stack([pose[:, 3:6], pose[:, 6:9], np.cross(pose[:, 3:6], pose[:, 6:9])], axis=1)  # the rotation; its third column is where the camera looks
            ahead = rows[:, :, 2]
            yaw = -np.arctan2(ahead[:, 0], ahead[:, 2])
            speed = np.linalg.norm(np.diff(pose[:, :3], axis=0, prepend=pose[:1, :3]), axis=1) * 15
            latent = np.zeros((len(pose), 32), dtype=np.float32)
            latent[:, 0] = np.sin(np.arange(len(pose)) * 0.8) * (speed > 0.1)
            valid = np.ones(len(pose), dtype=bool)
            if i == 1:
                valid[:400] = False
            np.savez(Path(folder) / "body" / path.name, latent=latent, pelvis=(pose[:, :3] - 0.3 * ahead).astype(np.float32), yaw=yaw.astype(np.float32), valid=valid)
        for steps in ("400", "800"):
            subprocess.run([
                sys.executable, "-m", "gray.motion", "--moments", str(Path(folder) / "moments.npz"), "--tracks", str(paths["tracks"]), "--images", str(paths["images"]),
                "--cache", str(paths["cache"]), "--body", str(Path(folder) / "body"), "--save-dir", str(out), "--resume", "--steps", steps, "--eval-every", "400",
                "--batch-size", "32", "--width", "96", "--layers", "2", "--heads", "4", "--workers", "0", "--val-moments", "64", "--warmup", "50", "--lr", "1e-3", "--sample-steps", "8",
            ], cwd=ROOT, check=True)
        log = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
        assert [entry["step"] for entry in log] == [400, 800]
        final = log[-1]
        print({key: round(value, 2) for key, value in final.items()})
        assert abs(final["under_cm"] - 30) < 1, "the made-up pelvis should be 30 cm from under the camera"
        assert final["warm_pelvis_cm"] < 10 and final["cold_pelvis_cm"] < 10, "the pelvis does not follow the given head path"
        assert final["warm_yaw_deg"] < 10
        assert (out / "best.pt").is_file()
        print("ok")
