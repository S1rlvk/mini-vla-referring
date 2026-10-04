"""Verifiers v1 taskset for coordinate grounding under robot self-occlusion."""

from __future__ import annotations

import json
import math
import re

import verifiers.v1 as vf
from pydantic import Field

from .scene import make_scene


class CornerData(vf.TaskData):
    target_x: float
    target_y: float
    condition: str


class CornerConfig(vf.TaskConfig):
    success_radius: float = Field(default=0.035, gt=0, le=0.25)


class ClothCornerTask(vf.Task[CornerData, vf.State, CornerConfig]):
    @vf.metric
    def pixel_error(self, trace) -> dict[str, float]:
        point = _parse_point(trace.last_reply)
        error = 1.0 if point is None else math.hypot(
            point[0] - self.data.target_x, point[1] - self.data.target_y
        )
        return {"pixel_error": error, f"pixel_error/{self.data.condition}": error}

    @vf.metric
    def success(self, trace) -> dict[str, float]:
        point = _parse_point(trace.last_reply)
        if point is None:
            value = 0.0
        else:
            error = math.hypot(point[0] - self.data.target_x, point[1] - self.data.target_y)
            value = float(error <= self.config.success_radius)
        return {"success": value, f"success/{self.data.condition}": value}

    @vf.reward
    def coordinate_reward(self, trace) -> float:
        point = _parse_point(trace.last_reply)
        if point is None:
            return 0.0
        error = math.hypot(point[0] - self.data.target_x, point[1] - self.data.target_y)
        return max(0.0, 1.0 - error / 0.2)


class ClothOcclusionConfig(vf.TasksetConfig):
    n_examples: int = Field(default=200, ge=2, le=10000, multiple_of=2)
    seed: int = 17
    coordinate_decimals: int = Field(default=2, ge=2, le=6)


class ClothOcclusionTaskset(vf.Taskset[ClothCornerTask, ClothOcclusionConfig]):
    def load(self) -> list[ClothCornerTask]:
        tasks = []
        for pair in range(self.config.n_examples // 2):
            seed = self.config.seed + pair
            for occluded in (False, True):
                idx = len(tasks)
                image, x, y = make_scene(seed, occluded=occluded)
                condition = "self_occluded" if occluded else "visible_control"
                example_coordinate = f'{0.5:.{self.config.coordinate_decimals}f}'
                prompt = [vf.UserMessage(content=[
                    vf.TextContentPart(text=(
                        "This is an overhead view of a blue cloth on a table. Find the orange "
                        "marker at one cloth corner; the robot arm may hide it. Return exactly "
                        f"one JSON object, no list or markdown, with at least "
                        f"{self.config.coordinate_decimals} decimal places: "
                        f'{{"x":{example_coordinate},"y":{example_coordinate}}}. '
                        "Coordinates range from 0 at the top/left to 1 at the bottom/right."
                    )),
                    vf.ImageUrlContentPart(image_url=vf.ImageUrlSource(url=image)),
                ])]
                data = CornerData(idx=idx, name=f"corner-{condition}-{idx}", prompt=prompt,
                                  target_x=x, target_y=y, condition=condition)
                tasks.append(ClothCornerTask(data))
        return tasks


def _parse_point(reply: str) -> tuple[float, float] | None:
    match = re.search(r"\{[^{}]*\}", reply)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        x, y = float(parsed["x"]), float(parsed["y"])
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None
    if not (0 <= x <= 1 and 0 <= y <= 1):
        return None
    return x, y
