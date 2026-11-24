#!/usr/bin/env python3
"""The picture a model was given: the pinhole view of one frame of a recording, cut from its video as gray/pinhole.py does.

Needs the recording's video (<videos>/<sequence id>/preview_rgb.mp4), ffmpeg, and the lens of its camera. On the cluster the
lens is read from the recording; elsewhere from a file ``lens.json`` ({sequence id: {"lens": [15 numbers], "clockwise": bool}}).
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
import torch

from gray.rectify import lookup, rectify, video_frame

VIEW = 79.6  # degrees, as gray/pinhole.py
RATE = 15.0


class Seen:
    def __init__(self, videos: Path, lens: Path | None = None) -> None:
        self.videos, self.where = Path(videos), {}
        self.lens = json.loads(Path(lens).read_text()) if lens else None

    def view(self, name: str, frame: int, side: int = 448) -> np.ndarray:
        """(side, side, 3) u8."""
        video = self.videos / name / "preview_rgb.mp4"
        full = torch.from_numpy(video_frame(video, frame / RATE).copy()).permute(2, 0, 1)[None].float()
        key = (name, side, full.shape[-1])
        if key not in self.where:
            if self.lens is not None:
                lens, clockwise = np.array(self.lens[name]["lens"]), bool(self.lens[name]["clockwise"])
            else:
                from gray.rectify import calibration, mounting

                lens, on_glasses = calibration(self.videos / name)
                clockwise = bool(mounting(self.videos / name, on_glasses)[0] > 0)
            self.where[key] = lookup(lens, full.shape[-1], clockwise, view=VIEW, wide=2 * side, high=2 * side).float()
        return torch.nn.functional.avg_pool2d(rectify(full, self.where[key]), 2)[0].permute(1, 2, 0).round().clamp(0, 255).byte().numpy()

    def jpeg(self, name: str, frame: int, side: int = 448) -> str:
        from PIL import Image

        out = io.BytesIO()
        Image.fromarray(self.view(name, frame, side)).save(out, format="JPEG", quality=88)
        return base64.b64encode(out.getvalue()).decode()
