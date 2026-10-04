"""Verifiers v1 tasks for a kinematic 2D gripper reaching a moving cloth corner."""
from __future__ import annotations
import math
import random
import verifiers.v1 as vf
from pydantic import ConfigDict, Field
from PIL import Image, ImageDraw
from .scene import SIZE
from .temporal_scene import draw_cloth, encode

class ReachData(vf.TaskData):
    model_config = ConfigDict(frozen=True, extra='forbid')
    seed: int
    occluded: bool

class ReachState(vf.State):
    model_config = ConfigDict(extra='forbid')
    step: int = 0
    gripper: tuple[float,float] = (0.15,0.85)
    success: bool = False
    path_length_pixels: float = 0
    chunk_attempts: int = 0
    valid_chunks: int = 0

class ReachConfig(vf.TaskConfig):
    model_config = ConfigDict(extra='forbid')
    max_steps: int = Field(default=16,ge=1,le=100)
    max_delta: float = Field(default=0.05,gt=0,le=0.2)
    success_radius_pixels: float = Field(default=8.96,gt=0)

class ReachTask(vf.Task[ReachData,ReachState,ReachConfig]):
    def geometry(self, step):
        rng=random.Random(self.data.seed)
        cx,cy=rng.uniform(115,145),rng.uniform(110,140)
        hw,hh=rng.uniform(38,50),rng.uniform(30,42)
        angle=rng.uniform(-0.4,0.4)
        target_idx=rng.randrange(4)
        # Smooth exogenous rigid movement; no cloth contact or deformation.
        cx+=10*math.sin(step*0.18)
        cy+=8*math.sin(step*0.13)
        angle+=0.12*math.sin(step*0.16)
        corners=[(cx+sx*hw*math.cos(angle)-sy*hh*math.sin(angle),
                  cy+sx*hw*math.sin(angle)+sy*hh*math.cos(angle))
                 for sx,sy in [(-1,-1),(1,-1),(1,1),(-1,1)]]
        return corners,corners[target_idx]

    def error(self,state):
        _,target=self.geometry(state.step)
        return math.dist((state.gripper[0]*SIZE,state.gripper[1]*SIZE),target)

    def observe(self,state):
        image=Image.new('RGB',(SIZE,SIZE),(220,218,205))
        draw=ImageDraw.Draw(image)
        draw.rectangle((12,12,244,244),outline=(118,112,98),width=3)
        corners,target=self.geometry(state.step)
        image=draw_cloth(image,corners,target)
        draw=ImageDraw.Draw(image)
        x,y=state.gripper[0]*SIZE,state.gripper[1]*SIZE
        if self.data.occluded:
            draw.line(((22,234),(x,y)),fill=(75,78,83),width=22)
            draw.ellipse((x-9,y-9,x+9,y+9),fill=(49,52,56))
        # Red outline identifies current gripper location in either camera condition.
        draw.ellipse((x-6,y-6,x+6,y+6),outline=(220,35,40),width=2)
        return encode(image)

    def advance(self,state,action):
        dx,dy=action
        old=state.gripper
        point=(max(.06,min(.94,old[0]+dx)),max(.06,min(.94,old[1]+dy)))
        state.gripper=point
        state.step+=1
        state.path_length_pixels+=math.dist(old,point)*SIZE
        state.success=self.error(state)<=self.config.success_radius_pixels
        return state.success or state.step>=self.config.max_steps

    @vf.metric
    def endpoint_error_pixels(self,trace):
        return self.error(trace.state)

    @vf.reward
    def reach_success(self,trace):
        return float(trace.state.success)

    @vf.metric
    def valid_chunk_rate(self,trace):
        return trace.state.valid_chunks / max(1,trace.state.chunk_attempts)

class ChunkPilotConfig(vf.TasksetConfig):
    model_config=ConfigDict(extra='forbid')
    n_seeds: int = Field(default=4,ge=1,le=100)
    seed: int = 2001
    horizon: int = Field(default=8,ge=1,le=32)
    execute: int = Field(default=2,ge=1,le=32)
    max_steps: int = Field(default=16,ge=1,le=100)
    max_delta: float = Field(default=.05,gt=0,le=.2)

class ChunkReachTaskset(vf.Taskset[ReachTask,ChunkPilotConfig]):
    def load(self):
        if self.config.execute>self.config.horizon:
            raise ValueError('execute must not exceed horizon')
        return [ReachTask(ReachData(idx=2*i+int(occluded),seed=self.config.seed+i,
                                   occluded=occluded,name=f'reach-{self.config.seed+i}-{occluded}'),
                          ReachConfig(max_steps=self.config.max_steps,max_delta=self.config.max_delta))
                for i in range(self.config.n_seeds) for occluded in (False,True)]
