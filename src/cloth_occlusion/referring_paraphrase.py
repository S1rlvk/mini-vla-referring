"""Held-out phrasing test: same scenes/targets, expressions reworded outside the parser's grammar. Qwen points, oracle head reaches."""
from __future__ import annotations
import argparse, json, tomllib
from pathlib import Path
import numpy as np
from .referring_experiment import Config, build_split, rollout, summarize, atomic_json
from .referring_scene import frame, parse
from .referring_point import ask
from .trained_chunk import TrainConfig, Head

SHAPE={'circle':'disc','square':'block','triangle':'wedge'}
SIZE={'small':'tiny','large':'big'}
REL=[{'left of':'to the left of','right of':'to the right of','behind':'further back than','in front of':'closer to the viewer than'},
     {'left of':'on the left side of','right of':'on the right side of','behind':'above','in front of':'below'}]

def noun(a,style):
    shape=SHAPE.get(a.shape,'thing') if a.shape else 'thing'
    parts=[SIZE[a.size]] if a.size else []
    if a.color: parts.append(a.color)
    parts.append(shape)
    return ' '.join(parts)

def paraphrase(e,i):
    s=i%2; a=e.attributes
    if e.family=='attribute': return f'the {noun(a,s)}' if s==0 else f'that {noun(a,s)} over there'
    if e.family in ('relation','nested'): return f'the {noun(a,s)} {REL[s][e.relation]} the {noun(e.anchor,s)}'
    if e.family=='edge_context':
        edge={'top':'upper','bottom':'lower'}[e.relation]
        return f'the {noun(a,s)} in the {edge} part of the scene' if s==0 else f'of the {noun(a,s)}s, the one nearest the {e.relation} edge'
    if e.family=='not_red':
        return f'the {SHAPE[a.shape]} whose color is anything but red' if s==0 else f"pick the {SHAPE[a.shape]} that isn't red"
    return (f'the {noun(a,s)} in the middle of the {noun(e.anchor,s)} and the {noun(e.anchor2,s)}' if s==0
            else f'the {noun(a,s)} halfway between the {noun(e.anchor,s)} and the {noun(e.anchor2,s)}')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--heads',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--limit',type=int,default=0)
    a=ap.parse_args(); c=Config.model_validate(tomllib.loads(a.config.read_text())); a.output.mkdir(parents=True,exist_ok=True)
    head=Head(TrainConfig(feature_width=72,epochs=1,batch_size=128,learning_rate=.001,random_seed=c.random_seed)); head.load(a.heads/'oracle_head.npz')
    norm=np.load(a.heads/'oracle_normalization.npz'); variant={'head':head,'mean':norm['mean'],'scale':norm['scale']}
    from mlx_vlm import load
    model,processor=load(c.model_path); cfg=json.loads((Path(c.model_path)/'config.json').read_text())
    cases,_=build_split(c,'test')
    if a.limit: cases=cases[::max(1,len(cases)//a.limit)][:a.limit]
    rows=[]; parsed=0; target_err=[]
    with (a.output/'rollouts.jsonl').open('w') as stream:
        for i,data in enumerate(cases):
            text=paraphrase(data.expression,i)
            try: parse(text); parsed+=1
            except ValueError: pass
            answer,point=ask(model,processor,cfg,frame(data.objects,(.5,.5)),text)
            target=next(o for o in data.objects if o.id==data.target_id)
            if point is None: point=(.5,.5)
            else: point=(min(max(point[0],.06),.94),min(max(point[1],.06),.94))
            err=float(np.hypot(point[0]-target.x,point[1]-target.y))*256; target_err.append(err)
            row=rollout(c,data,'oracle',variant,None,aim=point,label='qwen_point')
            row.update(paraphrase=text,original=data.expression.text,raw_answer=answer,point=point,point_error_pixels=err,
                       point_in_target=err<=target.radius*256)
            rows.append(row); stream.write(json.dumps(row)+'\n'); stream.flush()
            print(f'{i+1}/{len(cases)} {data.expression.level} "{text}" err={err:.0f}px success={row["success"]}',flush=True)
    out={'summary':summarize(rows).get('qwen_point',{}),'parser_coverage':f'{parsed}/{len(cases)}',
         'point_error_median_px':float(np.median(target_err)),'point_in_target_fraction':float(np.mean([r['point_in_target'] for r in rows]))}
    atomic_json(a.output/'comparison.json',out); print(json.dumps(out,indent=1))
if __name__=='__main__': main()
