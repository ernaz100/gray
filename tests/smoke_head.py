"""Run gray/head.py end to end on tiny made-up data (CPU, about a minute).

    python tests/smoke_head.py

The made-up world: every 5 s a sentence begins. After a "go" sentence the head walks straight
ahead at 1 m/s for the 5 s, after a "stay" sentence it stands still; the two kinds of sentence
have different text features. All images look alike and nothing before now tells which comes next, so a model can only be
right by reading the text: on the moments that go, one generated path must be far closer to the
recorded one than standing still is. The run is stopped half way and resumed.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gray.moments import FUTURE  # noqa: E402


def make(folder: Path, recordings: int = 40, sentences: int = 14) -> dict[str, Path]:
    rng = np.random.default_rng(0)
    paths = {name: folder / name for name in ("tracks", "images", "cache")}
    rows = {key: [] for key in ("sequence", "sentence", "now", "reach", "going", "held")}
    word = rng.choice([-1.0, 1.0], size=768)  # "go" is this pattern, "stay" its negative
    view = rng.normal(size=384)  # every frame looks the same, so the images cannot be used to tell moments apart
    for r in range(recordings):
        name = f"recording{r}"
        yaw = rng.uniform(0, 2 * np.pi)
        ahead, right = np.array([-np.sin(yaw), 0.0, np.cos(yaw)]), np.array([np.cos(yaw), 0.0, np.sin(yaw)])
        axes = np.stack([np.cross(right, ahead), right, ahead], axis=1)  # camera axes in columns
        goes = rng.random(sentences) < 0.5
        count = 60 + sentences * FUTURE + 40
        position = np.zeros((count, 3))
        position[:, 1] = 1.6
        for s in range(sentences):
            begin = 60 + s * FUTURE
            if goes[s]:
                position[begin + 1 :] += ahead / 15 * np.minimum(np.arange(1, count - begin), FUTURE)[:, None]
            for key, value in (("sequence", name), ("sentence", s), ("now", begin), ("reach", 5.0 * goes[s]), ("going", goes[s]), ("held", r % 4 == 0)):
                rows[key].append(value)
        pose = np.concatenate([position, np.tile(np.concatenate([axes[0], axes[1]]), (count, 1))], axis=1).astype(np.float32)
        for path in paths.values():
            (path / name).mkdir(parents=True, exist_ok=True)
        np.savez(paths["tracks"] / f"{name}.npz", pose=pose, floor=np.zeros(count, np.float32), valid=np.ones(count, bool),
                 sentences=np.array([[60 + s * FUTURE, 60 + (s + 1) * FUTURE] for s in range(sentences)], dtype=np.int32))
        np.save(paths["images"] / name / "dino.npy", (view + 0.01 * rng.normal(size=(count, 384))).astype(np.float16))
        tokens = rng.normal(size=(sentences, 77, 768)) * 0.1 + np.where(goes, 1.0, -1.0)[:, None, None] * word
        np.save(paths["cache"] / name / "text_tokens.npy", tokens.astype(np.float16))
        np.save(paths["cache"] / name / "text_lengths.npy", rng.integers(4, 12, size=sentences).astype(np.int16))
    total = len(rows["now"])
    np.savez(folder / "moments.npz", **{key: np.array(value) for key, value in rows.items()}, subset=np.ones(total, bool), walk=np.array(rows["going"]),
             destination=np.array([""] * total), has_map=np.zeros(total, bool), floor=np.full(total, np.nan), wall=np.full(total, np.nan))
    return paths


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as folder:
        paths, out = make(Path(folder)), Path(folder) / "out"
        for steps in ("300", "600"):
            subprocess.run([
                sys.executable, "-m", "gray.head", "--moments", str(Path(folder) / "moments.npz"), "--tracks", str(paths["tracks"]), "--images", str(paths["images"]),
                "--cache", str(paths["cache"]), "--save-dir", str(out), "--resume", "--steps", steps, "--eval-every", "300", "--batch-size", "32", "--width", "96",
                "--layers", "2", "--heads", "4", "--workers", "0", "--val-moments", "64", "--warmup", "50", "--lr", "1e-3", "--sample-steps", "8",
            ], cwd=ROOT, check=True)
        log = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
        assert [entry["step"] for entry in log] == [300, 600]
        final = log[-1]
        print({key: round(value, 3) for key, value in final.items()})
        assert final["going_one_draw_cm"] < 0.5 * final["going_stay_cm"], "the generated path does not follow the text"
        assert final["one_draw_cm"] < final["stay_cm"]
        print("ok")
        for extra in (["--explore", "3"], ["--explore", "3", "--one-step"]):  # explorative training, as a flow model and in one step
            out = Path(folder) / ("out" + "".join(extra))
            subprocess.run([
                sys.executable, "-m", "gray.head", "--moments", str(Path(folder) / "moments.npz"), "--tracks", str(paths["tracks"]), "--images", str(paths["images"]),
                "--cache", str(paths["cache"]), "--save-dir", str(out), "--steps", "600", "--eval-every", "600", "--batch-size", "32", "--width", "96",
                "--layers", "2", "--heads", "4", "--workers", "0", "--val-moments", "64", "--warmup", "50", "--lr", "1e-3", "--sample-steps", "8", *extra,
            ], cwd=ROOT, check=True)
            final = json.loads((out / "log.jsonl").read_text().splitlines()[-1])
            print(extra, {key: round(value, 1) for key, value in final.items() if key.endswith("_cm")})
            assert final["going_one_draw_cm"] < 0.5 * final["going_stay_cm"], f"{extra}: the generated path does not follow the text"
        print("ok, also with exploration")
