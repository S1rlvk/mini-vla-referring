"""Supervised bounded chunk head, observed-marker frontend, and matched r calibration.

Qwen is not used by this diagnostic baseline. Expert target states are labels only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import time
import tomllib
from collections import defaultdict
from pathlib import Path
import numpy as np
from pydantic import BaseModel,ConfigDict,Field,model_validator
from PIL import ImageDraw
from .chunk_control import ReachTask,ReachData,ReachState,ReachConfig
from .chunk_eval import parse_chunk
from .temporal_scene import decode

class TrainConfig(BaseModel):
    model_config=ConfigDict(extra='forbid')
    horizon:int=Field(default=8,ge=1,le=32)
    max_delta:float=Field(default=.05,gt=0,le=.2)
    max_steps:int=Field(default=16,ge=1,le=100)
    train_seed:int=10000
    train_scenes:int=Field(default=320,ge=1)
    validation_seed:int=14000
    validation_scenes:int=Field(default=64,ge=1)
    calibration_seed:int=2001
    calibration_scenes:int=Field(default=64,ge=1)
    confirmation_seed:int=3001
    confirmation_scenes:int=Field(default=64,ge=1)
    stress_confirmation_seed:int=4001
    stress_confirmation_scenes:int=Field(default=64,ge=1)
    epochs:int=Field(default=100,ge=1)
    batch_size:int=Field(default=256,ge=1)
    hidden_width:int=Field(default=128,ge=8)
    feature_width:int=Field(default=8,ge=8)
    learning_rate:float=Field(default=.001,gt=0)
    random_seed:int=42
    execution_intervals:list[int]=Field(default_factory=lambda:[1,2,4,8])

    @model_validator(mode='after')
    def check(self):
        if any(r<1 or r>self.horizon for r in self.execution_intervals):
            raise ValueError('Invalid execution interval')
        ranges=[set(range(getattr(self,k+'_seed'),getattr(self,k+'_seed')+getattr(self,k+'_scenes')))
                for k in ('train','validation','calibration','confirmation','stress_confirmation')]
        if any(ranges[i]&ranges[j] for i in range(len(ranges)) for j in range(i+1,len(ranges))):
            raise ValueError('Scene splits overlap')
        return self

class Tracker:
    def __init__(self):
        self.position=np.array([.5,.5],dtype=np.float64)
        self.velocity=np.zeros(2)
        self.last_step=None

    def features(self,image,gripper,step,max_delta):
        rgb=np.asarray(image)
        mask=(rgb[:,:,0]>210)&(rgb[:,:,1]>90)&(rgb[:,:,1]<160)&(rgb[:,:,2]<70)
        ys,xs=np.nonzero(mask)
        visible=len(xs)>0
        if visible:
            point=np.array([xs.mean(),ys.mean()])/256
            if self.last_step is not None and step>self.last_step:
                self.velocity=(point-self.position)/(step-self.last_step)
            self.position=point;self.last_step=step
        age=step-(self.last_step if self.last_step is not None else 0)
        estimate=np.clip(self.position+self.velocity*age,.02,.98)
        features=np.concatenate((2*np.asarray(gripper)-1,2*estimate-1,
                    np.clip(self.velocity/max_delta,-1,1),[float(visible)*2-1,min(age/8,1)]))
        return features,visible,estimate


def observation(task,state,stress=False):
    image=decode(task.observe(state))
    # Fixed band independent of the target; controlled observation stress only.
    # First frame clear supplies identity. This mask is not physical self-occlusion.
    phase=(state.step+(task.data.seed%8 if stress=='phase_shifted' else 0))%8
    if stress and state.step>0 and phase in (2,3,4,5):
        ImageDraw.Draw(image).rectangle((40,85,220,180),fill=(75,78,83))
    return image

class Head:
    def __init__(self,c):
        rng=np.random.default_rng(c.random_seed)
        self.c=c
        dims=(c.feature_width,c.hidden_width,c.hidden_width,c.horizon*2)
        self.params=[]
        for a,b in zip(dims[:-1],dims[1:]):
            self.params.extend([rng.normal(0,math.sqrt(1/a),(a,b)),np.zeros(b)])
        self.m=[np.zeros_like(p) for p in self.params];self.v=[np.zeros_like(p) for p in self.params];self.update=0

    def forward(self,x):
        w1,b1,w2,b2,w3,b3=self.params
        h1=np.tanh(x@w1+b1);h2=np.tanh(h1@w2+b2);y=np.tanh(h2@w3+b3)
        return y,(x,h1,h2)

    def actions(self,features):
        y,_=self.forward(features[None,:])
        return (y.reshape(self.c.horizon,2)*self.c.max_delta).tolist()

    def train_batch(self,x,target):
        output,(x,h1,h2)=self.forward(x)
        gradient=2*(output-target)/output.size*(1-output**2)
        w1,b1,w2,b2,w3,b3=self.params
        g3=h2.T@gradient;gb3=gradient.sum(0)
        d2=(gradient@w3.T)*(1-h2**2)
        g2=h1.T@d2;gb2=d2.sum(0)
        d1=(d2@w2.T)*(1-h1**2)
        grads=[x.T@d1,d1.sum(0),g2,gb2,g3,gb3]
        self.update+=1
        for p,m,v,g in zip(self.params,self.m,self.v,grads):
            m[:]=.9*m+.1*g;v[:]=.999*v+.001*g*g
            p-=self.c.learning_rate*(m/(1-.9**self.update))/(np.sqrt(v/(1-.999**self.update))+1e-8)
        return float(np.mean((output-target)**2))

    def save(self,path):
        np.savez_compressed(path,**{f'p{i}':p for i,p in enumerate(self.params)})

    def load(self,path):
        with np.load(path) as data:
            for i,p in enumerate(self.params):p[:]=data[f'p{i}']


def dataset(c,seed,count):
    rng=np.random.default_rng(seed)
    xs=[];ys=[]
    for scene in range(seed,seed+count):
        for stress in (False,True):
            task=ReachTask(ReachData(seed=scene,occluded=True),ReachConfig(max_steps=c.max_steps,max_delta=c.max_delta))
            tracker=Tracker()
            for step in range(c.max_steps):
                point=tuple(rng.uniform(.06,.94,2))
                state=ReachState(step=step,gripper=point)
                features,_,_=tracker.features(observation(task,state,stress),point,step,c.max_delta)
                # Expert can see future simulator states, only for supervised action labels.
                pos=np.array(point);actions=[]
                for offset in range(1,c.horizon+1):
                    _,target=task.geometry(step+offset)
                    move=np.clip(np.asarray(target)/256-pos,-c.max_delta,c.max_delta)
                    actions.append(move/c.max_delta);pos+=move
                xs.append(features);ys.append(np.asarray(actions).flatten())
    return np.asarray(xs),np.asarray(ys)


def rollout(head,c,seed,occluded,r,stress=False):
    task=ReachTask(ReachData(seed=seed,occluded=occluded),ReachConfig(max_steps=c.max_steps,max_delta=c.max_delta))
    state=ReachState();tracker=Tracker();chunks=[]
    while state.step<c.max_steps and not state.success:
        start=time.perf_counter()
        features,visible,estimate=tracker.features(observation(task,state,stress),state.gripper,state.step,c.max_delta)
        actions=head.actions(features)
        valid=True;error=None
        try: actions=parse_chunk(json.dumps({'actions':actions}),c.horizon,c.max_delta)
        except (ValueError,TypeError,KeyError) as exc:
            valid=False;error=str(exc);actions=[(0.,0.)]*c.horizon
        state.chunk_attempts+=1
        state.valid_chunks+=int(valid)
        latency=time.perf_counter()-start
        chunk={'step_before':state.step,'gripper_before':state.gripper,'marker_visible':visible,
               'estimated_target':estimate.tolist(),'valid':valid,'invalid_reason':error,
               'predicted_actions':actions,'executed_actions':[],'latency_seconds':latency}
        for action in actions[:r]:
            done=task.advance(state,action);chunk['executed_actions'].append(action)
            if done:break
        chunk.update(step_after=state.step,error_after_pixels=task.error(state),gripper_after=state.gripper)
        chunks.append(chunk)
    return {'seed':seed,'condition':('phase_shifted_band_stress' if stress=='phase_shifted' else ('fixed_band_stress' if stress else ('arm_rendered' if occluded else 'clear_view'))),
            'r':r,'success':state.success,'endpoint_error_pixels':task.error(state),'steps':state.step,'chunks':chunks}


def summarize(rows):
    groups=defaultdict(list)
    for row in rows:groups[(row['r'],row['condition'])].append(row)
    result={}
    for (r,condition),items in groups.items():
        chunks=[chunk for item in items for chunk in item['chunks']]
        result[f'r{r}/{condition}']={'n':len(items),'successes':sum(x['success'] for x in items),
            'success_rate':sum(x['success'] for x in items)/len(items),
            'valid_chunk_rate':sum(x['valid'] for x in chunks)/len(chunks),
            'valid_chunks':sum(x['valid'] for x in chunks),'chunk_count':len(chunks),
            'mean_endpoint_error_pixels':float(np.mean([x['endpoint_error_pixels'] for x in items])),
            'mean_calls':len(chunks)/len(items),'mean_steps':float(np.mean([x['steps'] for x in items])),
            'hidden_decision_rate':sum(not x['marker_visible'] for x in chunks)/len(chunks),
            'mean_inference_seconds':float(np.mean([x['latency_seconds'] for x in chunks]))}
    return result


def evaluate(head,c,seed,count,intervals):
    return [rollout(head,c,s,occluded,r,stress)
        for r in intervals for s in range(seed,seed+count)
        for occluded,stress in ((False,False),(True,False),(True,True))]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('trained_chunk.toml'))
    parser.add_argument('--output',type=Path,default=Path('runs/trained_chunk'))
    args=parser.parse_args()
    with args.config.open('rb') as f:c=TrainConfig.model_validate(tomllib.load(f))
    args.output.mkdir(parents=True,exist_ok=True)
    report={'config':c.model_dump(mode='json'),'qwen_used':False,'training':'supervised_behavior_cloning',
            'input':'image-derived marker tracker, history, gripper proprioception',
            'architecture':'8-input two-layer tanh MLP with bounded 8x2 output',
            'validity':'fixed shape and tanh bounds by construction; finite/bounds/shape checked before every execution',
            'physics':'2D kinematic rigid cloth; simulation paused during inference',
            'completed':False,'learning_curve':[]}
    head=Head(c);report['parameters']=sum(p.size for p in head.params)
    print('Generating train/validation observations and expert labels',flush=True)
    x,y=dataset(c,c.train_seed,c.train_scenes)
    vx,vy=dataset(c,c.validation_seed,c.validation_scenes)
    report.update(training_samples=len(x),validation_samples=len(vx))
    np.savez_compressed(args.output/'demonstrations.npz',train_x=x,train_y=y,validation_x=vx,validation_y=vy)
    untrained=evaluate(head,c,c.calibration_seed,c.calibration_scenes,[2])
    report['untrained_summary']=summarize(untrained)
    rng=np.random.default_rng(c.random_seed)
    best=float('inf');best_path=args.output/'head.npz'
    for epoch in range(c.epochs+1):
        if epoch:
            order=rng.permutation(len(x))
            for start in range(0,len(x),c.batch_size):
                batch=order[start:start+c.batch_size];head.train_batch(x[batch],y[batch])
        prediction,_=head.forward(vx);loss=float(np.mean((prediction-vy)**2))
        if loss<best:best=loss;head.save(best_path);report['best_epoch']=epoch
        if epoch in (0,5,10,25,50,75,c.epochs):
            validation_rollouts=evaluate(head,c,c.validation_seed,min(16,c.validation_scenes),[2])
            point={'epoch':epoch,'validation_action_mse':loss,'validation_task_summary':summarize(validation_rollouts)}
            report['learning_curve'].append(point)
            print(json.dumps(point),flush=True)
        (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    head.load(best_path)
    print('Sweeping matched calibration scenes',flush=True)
    calibration=evaluate(head,c,c.calibration_seed,c.calibration_scenes,c.execution_intervals)
    summary=summarize(calibration);report['calibration_summary']=summary
    # Choose only on calibration: maximize aggregate success, then minimize calls.
    choices=[]
    for r in c.execution_intervals:
        subset=[row for row in calibration if row['r']==r]
        choices.append((sum(row['success'] for row in subset)/len(subset),
                        -sum(len(row['chunks']) for row in subset)/len(subset),r))
    chosen=max(choices)[2];report['selected_r']=chosen
    confirmation=evaluate(head,c,c.confirmation_seed,c.confirmation_scenes,[chosen])
    report['confirmation_summary']=summarize(confirmation)
    report['head_sha256']=hashlib.sha256(best_path.read_bytes()).hexdigest()
    report['completed']=True
    (args.output/'rollouts.json').write_text(json.dumps({'untrained':untrained,'calibration':calibration,'confirmation':confirmation},indent=2)+'\n')
    (args.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'calibration':summary,'selected_r':chosen,'confirmation':report['confirmation_summary']},indent=2),flush=True)

if __name__=='__main__':main()
