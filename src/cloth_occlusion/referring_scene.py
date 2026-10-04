"""Image-grounded referring expressions and collision-aware 2D reaching (v1)."""
from __future__ import annotations
import heapq
import math
import random
import re
from types import MappingProxyType
from typing import Literal
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pydantic import BaseModel, ConfigDict, Field, model_validator
import verifiers.v1 as vf

COLORS = MappingProxyType({'red': (220,65,65), 'green': (55,175,85), 'blue': (55,100,220),
                          'yellow': (230,195,45), 'purple': (165,75,195)})
SHAPES = ('circle','square','triangle')

class Object(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    id: int
    color: str
    shape: Literal['circle','square','triangle']
    size: Literal['small','large']
    x: float
    y: float
    radius: float

class Attributes(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    color: str | None = None
    shape: str | None = None
    size: str | None = None
    not_color: str | None = None

class Expression(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True)
    text: str
    level: Literal['L1','L2','L3','L4']
    family: str
    attributes: Attributes
    relation: str | None = None
    anchor: Attributes | None = None
    anchor2: Attributes | None = None

def attribute_matches(obj, attr):
    return all(getattr(obj,k)==getattr(attr,k) for k in ('color','shape','size') if getattr(attr,k) is not None) and (attr.not_color is None or obj.color!=attr.not_color)

def resolve(objects, expression, margin=.06):
    candidates = [o for o in objects if attribute_matches(o,expression.attributes)]
    relation = expression.relation
    if relation in ('top','bottom'):
        if not candidates: return []
        values = sorted(candidates,key=lambda o:o.y,reverse=relation=='bottom')
        return [values[0].id] if len(values)==1 or abs(values[0].y-values[1].y)>.06 else []
    if relation:
        anchors = [o for o in objects if attribute_matches(o,expression.anchor)]
        if len(anchors)!=1: return []
        a = anchors[0]
        candidates = [o for o in candidates if o.id!=a.id]
        if relation=='between':
            other = [o for o in objects if attribute_matches(o,expression.anchor2)]
            if len(other)!=1 or other[0].id==a.id: return []
            b = other[0]; v=np.array([b.x-a.x,b.y-a.y]); vv=float(v@v)
            if vv<.01: return []
            selected=[]
            for o in candidates:
                if o.id==b.id: continue
                p=np.array([o.x-a.x,o.y-a.y]); t=float(p@v/vv)
                if .15<t<.85 and np.linalg.norm(p-t*v)<.10: selected.append(o.id)
            return selected
        predicates = {'left of':lambda o:o.x<a.x-margin, 'right of':lambda o:o.x>a.x+margin,
                      'behind':lambda o:o.y<a.y-margin, 'in front of':lambda o:o.y>a.y+margin}
        candidates = [o for o in candidates if predicates[relation](o)]
    return [o.id for o in candidates]

def label(attr):
    return ' '.join(x for x in (attr.size,attr.color,attr.shape or 'object') if x)

def expressions(objects, level, family=None):
    result=[]; seen=set()
    def add(e):
        match=resolve(objects,e)
        # A tolerance may enforce perceptual clearance but must never eliminate
        # a second match under the ordinary strict left/right/above/below reading.
        if len(match)==1 and resolve(objects,e,margin=0)==match and e.text not in seen:
            seen.add(e.text); result.append((e,match[0]))
    for o in objects:
        if level=='L1':
            a=Attributes(color=o.color,shape=o.shape)
            add(Expression(text=f'the {label(a)}',level=level,family='attribute',attributes=a))
        elif level in ('L2','L3'):
            a=Attributes(shape=o.shape) if level=='L2' else Attributes(size=o.size,color=o.color,shape=o.shape)
            # Relations must disambiguate, rather than merely decorate an already unique attribute.
            if sum(attribute_matches(p,a) for p in objects)>1:
                for anchor in objects:
                    b=Attributes(color=anchor.color,shape=anchor.shape)
                    for relation in ('left of','right of','behind','in front of'):
                        add(Expression(text=f'the {label(a)} {relation} the {label(b)}',level=level,
                                       family='relation' if level=='L2' else 'nested',attributes=a,relation=relation,anchor=b))
            if level=='L3':
                a=Attributes(color=o.color,shape=o.shape)
                if sum(attribute_matches(p,a) for p in objects)>1:
                    for edge in ('top','bottom'):
                        add(Expression(text=f'the {label(a)} near the {edge} edge',level=level,
                                       family='edge_context',attributes=a,relation=edge))
        elif family=='not_red':
            a=Attributes(shape=o.shape,not_color='red')
            # Negation must change the candidate set.
            if any(p.shape==o.shape and p.color=='red' for p in objects):
                add(Expression(text=f'the {o.shape} that is not red',level=level,family=family,attributes=a))
        else:
            a=Attributes(shape=o.shape)
            for i,anchor in enumerate(objects):
                for other in objects[i+1:]:
                    b=Attributes(color=anchor.color,shape=anchor.shape); c=Attributes(color=other.color,shape=other.shape)
                    add(Expression(text=f'the {label(a)} between the {label(b)} and the {label(c)}',level=level,
                                   family='between',attributes=a,relation='between',anchor=b,anchor2=c))
    return result

def draw_object(draw,obj,fill):
    x,y,r=obj.x*256,obj.y*256,obj.radius*256
    if obj.shape=='circle': draw.ellipse((x-r,y-r,x+r,y+r),fill=fill)
    elif obj.shape=='square': draw.rectangle((x-r,y-r,x+r,y+r),fill=fill)
    else: draw.polygon(((x,y-r),(x-r,y+r),(x+r,y+r)),fill=fill)

def object_masks(objects):
    result={}
    for obj in objects:
        image=Image.new('L',(256,256)); draw_object(ImageDraw.Draw(image),obj,255)
        result[obj.id]=np.asarray(image.filter(ImageFilter.MaxFilter(7)))>0
    return result

def frame(objects,gripper):
    image=Image.new('RGB',(256,256),(235,233,222)); draw=ImageDraw.Draw(image)
    draw.rectangle((8,8,248,248),outline=(145,140,130),width=2)
    for obj in objects: draw_object(draw,obj,COLORS[obj.color])
    x,y=gripper[0]*256,gripper[1]*256
    draw.line((x-4,y,x+4,y),fill=(15,20,25),width=2)
    draw.line((x,y-4,x,y+4),fill=(15,20,25),width=2)
    return image

def detect(image):
    """Palette blob segmentation, raster shape templates, observed centroid/size only."""
    rgb=np.asarray(image); found=[]
    for color,value in COLORS.items():
        mask=np.all(rgb==value,axis=2); visited=np.zeros(mask.shape,dtype=bool)
        for yy,xx in zip(*np.nonzero(mask)):
            if visited[yy,xx]: continue
            stack=[(int(xx),int(yy))]; visited[yy,xx]=True; pixels=[]
            while stack:
                x,y=stack.pop(); pixels.append((x,y))
                for nx,ny in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
                    if 0<=nx<256 and 0<=ny<256 and mask[ny,nx] and not visited[ny,nx]:
                        visited[ny,nx]=True; stack.append((nx,ny))
            if len(pixels)<20: continue
            p=np.asarray(pixels); xmin,ymin=p.min(0); xmax,ymax=p.max(0)
            x,y=(xmin+xmax)/512,(ymin+ymax)/512; radius=max(xmax-xmin,ymax-ymin)/512
            observed=np.zeros((256,256),dtype=bool); observed[p[:,1],p[:,0]]=True
            scores={}
            for shape in SHAPES:
                proto=Object(id=0,color=color,shape=shape,size='small',x=x,y=y,radius=radius)
                template=Image.new('L',(256,256)); draw_object(ImageDraw.Draw(template),proto,255)
                tm=np.asarray(template)>0
                scores[shape]=np.count_nonzero(tm&observed)/max(1,np.count_nonzero(tm|observed))
            found.append(Object(id=0,color=color,shape=max(scores,key=scores.get),
                                size='small' if radius<.043 else 'large',x=x,y=y,radius=radius))
    found.sort(key=lambda o:(o.x,o.y))
    found=[o.model_copy(update={'id':i}) for i,o in enumerate(found)]
    ys,xs=np.nonzero(np.all(rgb==(15,20,25),axis=2))
    if not len(xs): raise ValueError('Gripper marker missing from observation')
    return found,(float(xs.mean()/256),float(ys.mean()/256))

def parse(text):
    """Fixed, full template grammar; no ground-truth AST is consumed by this baseline."""
    colors='|'.join(COLORS); shapes='|'.join(SHAPES)
    attr=rf'(?:(small|large) )?(?:({colors}) )?({shapes}|object)'
    def attributes(groups):
        size,color,shape=groups; return Attributes(size=size,color=color,shape=None if shape=='object' else shape)
    neg=re.fullmatch(rf'the ({shapes}) that is not red',text)
    if neg: return Expression(text=text,level='L4',family='not_red',attributes=Attributes(shape=neg[1],not_color='red'))
    between=re.fullmatch(rf'the {attr} between the {attr} and the {attr}',text)
    if between:
        g=between.groups(); return Expression(text=text,level='L4',family='between',attributes=attributes(g[:3]),relation='between',anchor=attributes(g[3:6]),anchor2=attributes(g[6:9]))
    edge=re.fullmatch(rf'the {attr} near the (top|bottom) edge',text)
    if edge:
        g=edge.groups(); return Expression(text=text,level='L3',family='edge_context',attributes=attributes(g[:3]),relation=g[3])
    rel=re.fullmatch(rf'the {attr} (left of|right of|behind|in front of) the {attr}',text)
    if rel:
        g=rel.groups(); return Expression(text=text,level='L3' if g[0] else 'L2',family='relation',attributes=attributes(g[:3]),relation=g[3],anchor=attributes(g[4:]))
    simple=re.fullmatch(rf'the {attr}',text)
    if simple: return Expression(text=text,level='L1',family='attribute',attributes=attributes(simple.groups()))
    raise ValueError(f'Unsupported expression: {text}')

class GroundingData(vf.TaskData):
    model_config=ConfigDict(extra='forbid',frozen=True)
    seed:int
    objects:tuple[Object,...]
    expression:Expression
    target_id:int

    @model_validator(mode='after')
    def unique(self):
        if len(self.objects) not in (2,3,4): raise ValueError('Expected 2–4 objects')
        if resolve(self.objects,self.expression)!=[self.target_id] or resolve(self.objects,self.expression,margin=0)!=[self.target_id]:
            raise ValueError('Expression is not uniquely grounded under both margin and natural spatial semantics')
        return self

class GroundingState(vf.State):
    model_config=ConfigDict(extra='forbid')
    gripper:tuple[float,float]=(.5,.5)
    step:int=0
    reached_object_id:int|None=None
    chunk_attempts:int=0
    valid_chunks:int=0

class GroundingConfig(vf.TaskConfig):
    model_config=ConfigDict(extra='forbid')
    max_steps:int=16
    max_delta:float=.05

def contact(objects,masks,start,end):
    n=max(1,math.ceil(math.dist(start,end)*256*2))
    for t in np.linspace(0,1,n+1)[1:]:
        point=np.asarray(start)+(np.asarray(end)-start)*t
        x,y=np.clip(np.rint(point*256).astype(int),0,255)
        touched=[o.id for o in objects if masks[o.id][y,x]]
        if touched: return touched[0],tuple(float(v) for v in point)
    return None,end

class GroundingTask(vf.Task[GroundingData,GroundingState,GroundingConfig]):
    def observe(self,state): return frame(self.data.objects,state.gripper)

    def advance(self,state,action):
        end=tuple(np.clip(np.asarray(state.gripper)+action,.06,.94))
        reached,point=contact(self.data.objects,object_masks(self.data.objects),state.gripper,end)
        state.gripper=point; state.step+=1; state.reached_object_id=reached
        return reached is not None or state.step>=self.config.max_steps

    @vf.reward
    def success(self,trace): return float(trace.state.reached_object_id==self.data.target_id)

    @vf.metric
    def decoy_touch(self,trace): return float(trace.state.reached_object_id is not None and trace.state.reached_object_id!=self.data.target_id)

    @vf.metric
    def valid_chunk_rate(self,trace): return trace.state.valid_chunks/max(1,trace.state.chunk_attempts)

def make_scene(seed,level,index):
    rng=random.Random(seed)
    family=('between' if index%2==0 else 'not_red') if level=='L4' else (('nested' if index%2==0 else 'edge_context') if level=='L3' else None)
    for attempt in range(30000):
        n=4 if family=='nested' else rng.randint(2,4); objs=[]
        shared=(rng.choice(tuple(COLORS)),rng.choice(SHAPES)) if family=='nested' or rng.random()<.6 else None
        for i in range(n):
            color,shape=shared if shared and i<(3 if family=='nested' else 2) else (rng.choice(tuple(COLORS)),rng.choice(SHAPES))
            size=rng.choice(('small','large')); radius=(8 if size=='small' else 14)/256
            for _ in range(100):
                x,y=rng.uniform(.13,.87),rng.uniform(.13,.87)
                if math.dist((x,y),(.5,.5))>.12 and all(math.dist((x,y),(o.x,o.y))>.19 for o in objs): break
            else: break
            objs.append(Object(id=i,color=color,shape=shape,size=size,x=x,y=y,radius=radius))
        if len(objs)!=n: continue
        # Attributes and positions are drawn before constructing any referring expression.
        choices=expressions(objs,level,family)
        if level=='L3':
            choices=[(e,t) for e,t in choices if e.family==family]
            if family=='nested':
                # Both size and relation must be causally necessary. With <=4
                # objects, all four descriptive clauses cannot each require a
                # separate contrastive decoy, so do not claim that stronger test.
                choices=[(e,t) for e,t in choices if resolve(objs,e.model_copy(update={
                    'attributes':e.attributes.model_copy(update={'size':None})}))!=[t]]
        if family=='between':
            choices=[(e,t) for e,t in choices if resolve(objs,e.model_copy(update={
                'relation':None,'anchor':None,'anchor2':None}))!=[t]]
        targets=sorted({target for _,target in choices})
        if len(targets)<2: continue
        a,b=rng.sample(targets,2); pair=[]
        for target in (a,b):
            expression=rng.choice([e for e,t in choices if t==target])
            pair.append(GroundingData(seed=seed,objects=tuple(objs),expression=expression,target_id=target))
        return pair,attempt+1
    raise RuntimeError(f'Failed to generate counterfactual scene {seed}/{level}/{family}')

def expert(task,gripper,horizon=8):
    """Visibility graph around conservative decoy disks; privileged target for labels only."""
    target=next(o for o in task.data.objects if o.id==task.data.target_id)
    # Rasterization, 3 px gripper dilation, and collision rounding require a margin
    # beyond the continuous shape's circumscribed radius.
    obstacles=[(np.array([o.x,o.y]),o.radius*math.sqrt(2)+.03) for o in task.data.objects if o.id!=target.id]
    nodes=[np.asarray(gripper,dtype=float),np.array([target.x,target.y])]
    for center,radius in obstacles:
        for theta in np.linspace(0,2*math.pi,12,endpoint=False):
            p=center+radius*1.05*np.array([math.cos(theta),math.sin(theta)])
            if np.all((p>.06)&(p<.94)): nodes.append(p)
    def clear(a,b):
        v=b-a; vv=float(v@v)
        for center,radius in obstacles:
            t=np.clip(float((center-a)@v/max(vv,1e-12)),0,1)
            if np.linalg.norm(a+t*v-center)<radius: return False
        return True
    dist={0:0.}; parent={}; queue=[(0.,0)]
    while queue:
        cost,i=heapq.heappop(queue)
        if cost!=dist[i]: continue
        if i==1: break
        for j in range(len(nodes)):
            if j==i or not clear(nodes[i],nodes[j]): continue
            candidate=cost+float(np.linalg.norm(nodes[j]-nodes[i]))
            if candidate<dist.get(j,float('inf')):
                dist[j]=candidate; parent[j]=i; heapq.heappush(queue,(candidate,j))
    if 1 not in dist: raise ValueError('No collision-free expert route')
    indices=[1]
    while indices[-1]!=0: indices.append(parent[indices[-1]])
    waypoints=[nodes[i] for i in reversed(indices[:-1])]; pos=np.asarray(gripper,dtype=float); actions=[]
    masks=object_masks(task.data.objects); terminal=False
    for _ in range(horizon):
        if terminal: actions.append([0.,0.]); continue
        while len(waypoints)>1 and np.linalg.norm(waypoints[0]-pos)<1e-6: waypoints.pop(0)
        v=waypoints[0]-pos; move=v*min(1.,task.config.max_delta/max(float(np.max(np.abs(v))),1e-12))
        reached,point=contact(task.data.objects,masks,pos,pos+move)
        if reached is not None and reached!=task.data.target_id:
            raise ValueError(f'Expert route touches decoy: seed={task.data.seed} target={task.data.target_id} reached={reached} start={pos.tolist()} end={(pos+move).tolist()} waypoint={waypoints[0].tolist()}')
        actions.append(move.tolist()); pos=np.asarray(point); terminal=reached is not None
    return np.asarray(actions)

class GroundingTasksetConfig(vf.TasksetConfig):
    model_config=ConfigDict(extra='forbid')
    seed:int=21000
    scenes:int=32
    levels:tuple[str,...]=('L1','L2','L3')

class GroundingTaskset(vf.Taskset[GroundingTask,GroundingTasksetConfig]):
    def load(self):
        for li,level in enumerate(self.config.levels):
            for i in range(self.config.scenes):
                pair,_=make_scene(self.config.seed+li*1000+i,level,i)
                for data in pair: yield GroundingTask(data,GroundingConfig())
