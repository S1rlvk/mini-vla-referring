"""Qwen-as-pointer: Qwen answers the expression with pixel coordinates, the proven oracle head reaches them."""
from __future__ import annotations
import argparse, json, re, tempfile, tomllib
from pathlib import Path
import numpy as np
from .referring_experiment import Config, build_split, task_for, rollout, summarize, counterfactuals, atomic_json
from .referring_scene import frame
from .trained_chunk import TrainConfig, Head

PROMPT=('This is a top-down view of a workspace with several colored shapes. Behind means higher in the image; '
        'in front of means lower. Point to the object described, as x and y on a 0 to 1000 scale '
        '(x=0 left edge, x=1000 right edge, y=0 top edge, y=1000 bottom edge). '
        'Do not explain. Reply with exactly one line in the form (x=NNN, y=NNN). Description: ')

def ask(model,processor,config,image,text):
    from mlx_vlm import generate
    from mlx_vlm.prompt_utils import apply_chat_template
    rgb=np.asarray(image).copy(); rgb[np.all(rgb==(15,20,25),axis=2)]=(235,233,222)  # drop gripper marker, as in features
    from PIL import Image
    with tempfile.NamedTemporaryFile(suffix='.png') as f:
        Image.fromarray(rgb).save(f.name)
        prompt=apply_chat_template(processor,config,PROMPT+text,num_images=1,enable_thinking=False)
        out=generate(model,processor,prompt,image=[f.name],max_tokens=96,temperature=0.0,verbose=False)
    answer=out.text if hasattr(out,'text') else str(out)
    m=re.search(r'x\s*=\s*(-?\d+(?:\.\d+)?)\D+?y\s*=\s*(-?\d+(?:\.\d+)?)',answer)
    return answer,(float(m.group(1))/1000,float(m.group(2))/1000) if m else None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--heads',type=Path,required=True,help='folder with oracle_head.npz + oracle_normalization.npz')
    ap.add_argument('--output',type=Path,required=True); ap.add_argument('--limit',type=int,default=0)
    a=ap.parse_args(); c=Config.model_validate(tomllib.loads(a.config.read_text())); a.output.mkdir(parents=True,exist_ok=True)
    head=Head(TrainConfig(feature_width=72,epochs=1,batch_size=128,learning_rate=.001,random_seed=c.random_seed)); head.load(a.heads/'oracle_head.npz')
    norm=np.load(a.heads/'oracle_normalization.npz'); variant={'head':head,'mean':norm['mean'],'scale':norm['scale']}
    from mlx_vlm import load
    model,processor=load(c.model_path); cfg=json.loads((Path(c.model_path)/'config.json').read_text())
    cases,_=build_split(c,'test')
    if a.limit: cases=cases[::max(1,len(cases)//a.limit)][:a.limit]
    rows=[]; parse_fail=0; target_err=[]
    with (a.output/'rollouts.jsonl').open('w') as stream:
        for i,data in enumerate(cases):
            answer,point=ask(model,processor,cfg,frame(data.objects,(.5,.5)),data.expression.text)
            target=next(o for o in data.objects if o.id==data.target_id)
            if point is None: parse_fail+=1; point=(.5,.5)
            else: point=(min(max(point[0],.06),.94),min(max(point[1],.06),.94))
            err=float(np.hypot(point[0]-target.x,point[1]-target.y))*256; target_err.append(err)
            row=rollout(c,data,'oracle',variant,None,aim=point,label='qwen_point')
            row.update(raw_answer=answer,point=point,point_error_pixels=err,point_in_target=err<=target.radius*256)
            rows.append(row); stream.write(json.dumps(row)+'\n'); stream.flush()
            print(f'{i+1}/{len(cases)} {data.expression.level} ans={answer!r} err={err:.0f}px success={row["success"]}',flush=True)
    summary=summarize(rows).get('qwen_point',{})
    out={'summary':summary,'point_error_median_px':float(np.median(target_err)),'parse_failures':parse_fail,
         'point_in_target_fraction':float(np.mean([r['point_in_target'] for r in rows]))}
    atomic_json(a.output/'comparison.json',out); print(json.dumps(out,indent=1))
if __name__=='__main__': main()
