"""Checks of gray/frames.py on made-up head paths. Run: python tests/test_frames.py"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gray import frames  # noqa: E402


def rot_y(yaw: torch.Tensor) -> torch.Tensor:
    """Yaw about Y, as STEER's rot_yaw: a +z direction turned by yaw points along (-sin, 0, cos) after the inverse."""
    c, s, o, i = torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)
    return torch.stack([torch.stack([c, o, s], -1), torch.stack([o, i, o], -1), torch.stack([-s, o, c], -1)], -2)


def rot_x(pitch: torch.Tensor) -> torch.Tensor:
    c, s, o, i = torch.cos(pitch), torch.sin(pitch), torch.zeros_like(pitch), torch.ones_like(pitch)
    return torch.stack([torch.stack([i, o, o], -1), torch.stack([o, c, -s], -1), torch.stack([o, s, c], -1)], -2)


def paths(batch: int = 16, length: int = 120, pitch_range: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Random head paths in an arbitrary ground frame, and the yaw of each view direction."""
    torch.manual_seed(0)
    yaw = torch.cumsum(torch.randn(batch, length) * 0.1, dim=1) + torch.rand(batch, 1) * 6
    pitch = (torch.rand(batch, length) - 0.5) * pitch_range
    rotation = rot_y(-yaw) @ rot_x(pitch)  # camera +z points along (-sin yaw, ., cos yaw)
    position = torch.cumsum(torch.randn(batch, length, 3) * 0.05, dim=1) + torch.tensor([3.0, 1.6, -2.0])
    return torch.cat([position, rotation[..., 0, :], rotation[..., 1, :]], dim=-1), yaw


def test_heading() -> None:
    head, yaw = paths()
    difference = frames.heading(head[..., 3:]) - yaw
    assert torch.atan2(difference.sin(), difference.cos()).abs().max() < 1e-4


def test_anchor_puts_now_at_origin() -> None:
    head, _ = paths()
    now = torch.randint(0, head.shape[1], (head.shape[0],))
    moved = frames.anchor(head, now)
    here = moved[torch.arange(len(now)), now]
    assert here[:, [0, 2]].abs().max() < 1e-5  # on the origin
    assert torch.allclose(here[:, 1], head[torch.arange(len(now)), now, 1])  # height unchanged
    assert frames.heading(here[:, 3:]).abs().max() < 1e-4  # looking along +z
    rotation = frames.matrix(moved[..., 3:])
    assert torch.allclose(rotation @ rotation.transpose(-1, -2), torch.eye(3).expand_as(rotation), atol=1e-5)


def test_anchor_keeps_the_shape() -> None:
    head, _ = paths()
    moved = frames.anchor(head, torch.full((head.shape[0],), 40))
    distances = lambda h: (h[:, :, None, :3] - h[:, None, :, :3]).norm(dim=-1)  # torch.cdist is too coarse in float32
    assert torch.allclose(distances(moved), distances(head), atol=1e-5)
    relative = lambda h: frames.matrix(h[:, :1, 3:]).transpose(-1, -2) @ frames.matrix(h[..., 3:])
    assert torch.allclose(relative(moved), relative(head), atol=1e-5)


def test_anchor_twice() -> None:
    head, _ = paths()
    first, second = torch.full((head.shape[0],), 25), torch.full((head.shape[0],), 90)
    assert torch.allclose(frames.anchor(frames.anchor(head, first), second), frames.anchor(head, second), atol=1e-4)


def test_looking_down_uses_the_right_axis() -> None:
    yaw = torch.tensor([0.7])
    rotation = rot_y(-yaw) @ rot_x(torch.tensor([math.pi / 2 - 0.05]))  # 3 degrees from straight down or up
    six = torch.cat([rotation[..., 0, :], rotation[..., 1, :]], dim=-1)
    forward = rotation[..., 2]
    assert torch.hypot(forward[..., 0], forward[..., 2]) < frames.MIN_HORIZONTAL
    assert torch.isfinite(frames.heading(six)).all()


def test_anchor_root_follows_the_head() -> None:
    head, yaw = paths()
    # A pelvis 10 cm behind the camera, body facing where the camera looks.
    back = torch.stack([torch.sin(yaw), -torch.cos(yaw)], dim=-1) * 0.1
    root = torch.cat([head[..., [0, 2]] + back, torch.sin(yaw)[..., None], torch.cos(yaw)[..., None]], dim=-1)
    now = torch.randint(0, head.shape[1], (head.shape[0],))
    moved, moved_head = frames.anchor_root(root, head, now), frames.anchor(head, now)
    rows = torch.arange(len(now))
    assert torch.allclose(moved[rows, now], torch.tensor([0.0, -0.1, 0.0, 1.0]).expand(len(now), 4), atol=1e-4)
    assert torch.allclose((moved[..., :2] - moved_head[..., [0, 2]]).norm(dim=-1), torch.full(yaw.shape, 0.1), atol=1e-4)
    difference = torch.atan2(moved[..., 2], moved[..., 3]) - frames.heading(moved_head[..., 3:])
    assert torch.atan2(difference.sin(), difference.cos()).abs().max() < 1e-3


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_"):
            function()
            print("ok", name)
