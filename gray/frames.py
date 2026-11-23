"""The canonical frame, re-anchored at "now".

A head pose is nine numbers: the RGB camera's position (x, height, z) and its rotation as
the first two rows of the camera-to-frame matrix (the 6D form of STEER's cache). The frame
is Y-up, with its origin on the floor under the camera and +z along the camera's horizontal
view direction, at one chosen token. STEER's cache anchors every window at its first token;
``anchor`` moves a window to any other token, so the model always sees "now" at the origin.

Conventions follow STEER's ``egoprimo/preprocess.py`` and ``root.py``: a direction with yaw
psi points along (-sin psi, 0, cos psi); the camera looks along its +z, and its +y is the
wearer's right, which gives the heading when the view is within 17 degrees of vertical.
"""

from __future__ import annotations

import torch

MIN_HORIZONTAL = 0.3


def matrix(six: torch.Tensor) -> torch.Tensor:
    """(..., 6) -> rotation matrices (..., 3, 3), camera axes in columns."""
    first = torch.nn.functional.normalize(six[..., :3], dim=-1)
    second = six[..., 3:] - (first * six[..., 3:]).sum(-1, keepdim=True) * first  # generated values need not be a rotation yet
    second = torch.nn.functional.normalize(second, dim=-1)
    return torch.stack([first, second, torch.cross(first, second, dim=-1)], dim=-2)


def heading(six: torch.Tensor) -> torch.Tensor:
    """Yaw (...) of the horizontal view direction of a (..., 6) rotation."""
    rotation = matrix(six)
    forward, right = rotation[..., 2], rotation[..., 1]
    x = torch.where(torch.hypot(forward[..., 0], forward[..., 2]) >= MIN_HORIZONTAL, forward[..., 0], -right[..., 2])
    z = torch.where(torch.hypot(forward[..., 0], forward[..., 2]) >= MIN_HORIZONTAL, forward[..., 2], right[..., 0])
    return -torch.atan2(x, z)


def anchor(head: torch.Tensor, now: torch.Tensor) -> torch.Tensor:
    """Head poses (B, L, 9) in any ground frame -> the frame anchored at token ``now`` (B,)."""
    here = head[torch.arange(head.shape[0], device=head.device), now]  # (B, 9)
    yaw = heading(here[:, 3:])
    cosine, sine = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
    x, z = head[..., 0] - here[:, None, 0], head[..., 2] - here[:, None, 2]
    position = torch.stack([cosine * x + sine * z, head[..., 1], -sine * x + cosine * z], dim=-1)
    rotation = matrix(head[..., 3:])
    first = cosine[..., None] * rotation[..., 0, :] + sine[..., None] * rotation[..., 2, :]
    return torch.cat([position, first, rotation[..., 1, :]], dim=-1)


def anchor_root(root: torch.Tensor, head: torch.Tensor, now: torch.Tensor) -> torch.Tensor:
    """STEER's root channels (B, L, 4) -> the frame anchored at token ``now`` of the head path (B, L, 9).

    The channels are pelvis x and z in metres and the body yaw as sine and cosine, in the same
    ground frame as ``head``.
    """
    here = head[torch.arange(head.shape[0], device=head.device), now]
    yaw = heading(here[:, 3:])
    cosine, sine = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
    x, z = root[..., 0] - here[:, None, 0], root[..., 1] - here[:, None, 2]
    # sin(psi - yaw) and cos(psi - yaw)
    return torch.stack([cosine * x + sine * z, -sine * x + cosine * z,
                        root[..., 2] * cosine - root[..., 3] * sine, root[..., 3] * cosine + root[..., 2] * sine], dim=-1)
