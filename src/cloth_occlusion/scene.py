"""Procedural overhead scenes for the first cloth self-occlusion experiment.

This deliberately starts as a rendering benchmark, not cloth dynamics: it isolates
whether a model can infer a cloth corner hidden by the robot arm.
"""

from __future__ import annotations

import base64
import io
import math
import random

from PIL import Image, ImageDraw

SIZE = 256


def make_scene(seed: int, *, occluded: bool) -> tuple[str, float, float]:
    rng = random.Random(seed)
    image = Image.new("RGB", (SIZE, SIZE), (220, 218, 205))
    draw = ImageDraw.Draw(image)
    # Table boundary and subtle repeatable texture help camera calibration without
    # making the target itself a trivial fixed pixel.
    draw.rectangle((12, 12, 244, 244), outline=(118, 112, 98), width=3)
    for x in range(24, SIZE, 16):
        draw.line((x, 16, x, 240), fill=(216, 213, 201), width=1)
    for y in range(24, SIZE, 16):
        draw.line((16, y, 240, y), fill=(216, 213, 201), width=1)

    cx, cy = rng.uniform(100, 156), rng.uniform(100, 156)
    half_w, half_h = rng.uniform(38, 58), rng.uniform(30, 48)
    angle = rng.uniform(-0.55, 0.55)
    corners = []
    for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
        x, y = sx * half_w, sy * half_h
        corners.append((cx + x * math.cos(angle) - y * math.sin(angle),
                        cy + x * math.sin(angle) + y * math.cos(angle)))
    target = corners[rng.randrange(4)]
    draw.polygon(corners, fill=(91, 151, 179), outline=(38, 74, 94), width=3)
    # Fold-like seams make the object read as cloth, while remaining static.
    draw.line((corners[0], corners[2]), fill=(151, 190, 204), width=2)
    draw.line((corners[1], corners[3]), fill=(151, 190, 204), width=2)
    # Draw the marker before the arm. The arm layers below physically occlude it
    # in the positive condition; the visible control redraws it on top afterward.
    x, y = target
    draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(240, 125, 42), outline=(80, 40, 10))

    # Simplified arm silhouette: shoulder to wrist capsule and a two-finger gripper.
    # Its pose is identical across each matched image pair and covers the target.
    tx, ty = target
    wrist_x = tx + rng.uniform(-3, 3)
    wrist_y = ty + rng.uniform(-3, 3)
    shoulder = (rng.choice([22, 234]), rng.uniform(35, 220))
    draw.line((shoulder, (wrist_x, wrist_y)), fill=(75, 78, 83), width=22)
    draw.ellipse((shoulder[0] - 12, shoulder[1] - 12, shoulder[0] + 12, shoulder[1] + 12), fill=(57, 60, 64))
    draw.ellipse((wrist_x - 11, wrist_y - 11, wrist_x + 11, wrist_y + 11), fill=(49, 52, 56))
    # Fingers extend in a random direction; their geometry remains observable.
    phi = rng.uniform(-math.pi, math.pi)
    ux, uy = math.cos(phi), math.sin(phi)
    px, py = -uy, ux
    for sign in (-1, 1):
        start = (wrist_x + sign * 5 * px, wrist_y + sign * 5 * py)
        end = (wrist_x + 20 * ux + sign * 10 * px, wrist_y + 20 * uy + sign * 10 * py)
        draw.line((start, end), fill=(39, 42, 46), width=5)

    # The paired control differs only by revealing the marker above the same arm.
    if not occluded:
        x, y = target
        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(240, 125, 42), outline=(80, 40, 10))

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    return uri, target[0] / SIZE, target[1] / SIZE
