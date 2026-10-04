"""Cloth sequences with optional rigid motion and visibility-selected arm poses."""
from __future__ import annotations
import base64
import io
import math
import random
from dataclasses import dataclass
from PIL import Image, ImageDraw
from .scene import SIZE

@dataclass(frozen=True)
class Sequence:
    earlier: str
    current: str
    target: tuple[float, float]
    wrist: tuple[float, float]
    visibility: str
    wrist_distance_pixels: float
    marker_reference: tuple[float, float]
    earlier_target: tuple[float, float]
    displacement_pixels: float
    translation_pixels: tuple[float, float]
    rotation_radians: float
    transformed_marker_reference: tuple[float, float]

def encode(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()

def decode(uri: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(uri.split(',', 1)[1]))).convert('RGB')

def marker_center(image: Image.Image) -> tuple[float, float]:
    points = [(x, y) for y in range(SIZE) for x in range(SIZE)
              if (lambda rgb: rgb[0] > 210 and 90 < rgb[1] < 160 and rgb[2] < 70)(image.getpixel((x, y)))]
    if not points:
        raise ValueError('No visible orange marker')
    return tuple(sum(p[axis] for p in points) / len(points) / SIZE for axis in (0, 1))

def make_sequence(seed: int, *, hidden: bool, moving: bool = False) -> Sequence:
    rng = random.Random(seed)
    image = Image.new('RGB', (SIZE, SIZE), (220, 218, 205))
    draw = ImageDraw.Draw(image)
    draw.rectangle((12, 12, 244, 244), outline=(118, 112, 98), width=3)
    cx, cy = rng.uniform(105, 151), rng.uniform(105, 151)
    hw, hh, angle = rng.uniform(38, 58), rng.uniform(30, 48), rng.uniform(-0.55, 0.55)
    corners = [(cx + sx*hw*math.cos(angle)-sy*hh*math.sin(angle),
                cy + sx*hw*math.sin(angle)+sy*hh*math.cos(angle))
               for sx, sy in [(-1,-1),(1,-1),(1,1),(-1,1)]]
    target_idx = rng.randrange(4)
    old_target = corners[target_idx]
    earlier = draw_cloth(image.copy(), corners, old_target)
    dx, dy, rotation = 0.0, 0.0, 0.0
    if moving:
        motion_rng = random.Random(seed ^ 0xC107)
        for _ in range(10000):
            dx, dy = motion_rng.uniform(-24,24), motion_rng.uniform(-24,24)
            rotation = motion_rng.uniform(-0.30,0.30)
            moved_corners = [transform(p,(cx,cy),dx,dy,rotation) for p in corners]
            if (all(26 <= v <= 230 for p in moved_corners for v in p)
                    and math.dist(old_target,moved_corners[target_idx]) >= 18):
                corners = moved_corners
                break
        else:
            raise RuntimeError('Could not sample bounded cloth movement')
    tx, ty = corners[target_idx]
    image = draw_cloth(image, corners, (tx,ty))
    # Independent proposals; conditioning on coverage still creates correlations.
    # Endpoint distance prevents the old wrist-center shortcut by construction.
    for _ in range(20000):
        shoulder = (rng.choice([22,234]), rng.uniform(25,231))
        wrist = (rng.uniform(25,231), rng.uniform(25,231))
        if math.dist(wrist, (tx,ty)) < 35:
            continue
        mask = Image.new('L', (SIZE,SIZE))
        md = ImageDraw.Draw(mask)
        md.line((shoulder,wrist), fill=255, width=38)
        md.ellipse((wrist[0]-19,wrist[1]-19,wrist[0]+19,wrist[1]+19), fill=255)
        for offset in (-8,8):
            md.line((wrist[0]+offset,wrist[1],wrist[0]+offset,wrist[1]+18),fill=255,width=5)
        coverage = [mask.getpixel((round(tx)+dx, round(ty)+dy)) > 0
                    for dx in range(-8,9) for dy in range(-8,9)]
        if (hidden and all(coverage)) or (not hidden and not any(coverage)):
            break
    else:
        raise RuntimeError('Could not sample requested coverage')
    arm = Image.new('RGB', (SIZE,SIZE), (75,78,83))
    ad = ImageDraw.Draw(arm)
    ad.ellipse((wrist[0]-19,wrist[1]-19,wrist[0]+19,wrist[1]+19),fill=(49,52,56))
    for offset in (-8,8):
        ad.line((wrist[0]+offset,wrist[1],wrist[0]+offset,wrist[1]+18),fill=(39,42,46),width=5)
    current = Image.composite(arm,image,mask)
    reference = marker_center(earlier)
    mapped = transform((reference[0]*SIZE,reference[1]*SIZE),(cx,cy),dx,dy,rotation)
    return Sequence(encode(earlier), encode(current), (tx/SIZE,ty/SIZE),
                    (wrist[0]/SIZE,wrist[1]/SIZE), 'hidden' if hidden else 'visible',
                    math.dist(wrist,(tx,ty)), reference,
                    (old_target[0]/SIZE,old_target[1]/SIZE), math.dist(old_target,(tx,ty)),
                    (dx,dy), rotation, (mapped[0]/SIZE,mapped[1]/SIZE))

def transform(point, center, dx, dy, rotation):
    x, y = point[0]-center[0], point[1]-center[1]
    return (center[0]+dx+x*math.cos(rotation)-y*math.sin(rotation),
            center[1]+dy+x*math.sin(rotation)+y*math.cos(rotation))

def draw_cloth(image, corners, target):
    draw = ImageDraw.Draw(image)
    draw.polygon(corners, fill=(91,151,179), outline=(38,74,94), width=3)
    draw.line((corners[0], corners[2]), fill=(151,190,204), width=2)
    draw.line((corners[1], corners[3]), fill=(151,190,204), width=2)
    tx, ty = target
    draw.ellipse((tx-6,ty-6,tx+6,ty+6), fill=(240,125,42), outline=(80,40,10))
    return image
