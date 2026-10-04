"""Predeclared instruction-conditioned grounding ladder; frozen Qwen, matched MLPs."""
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
from pydantic import BaseModel, ConfigDict, Field, model_validator
from .referring_scene import (GroundingData,GroundingTask,GroundingState,GroundingConfig,
                              make_scene,detect,parse,resolve,expert,frame)
from .trained_chunk import TrainConfig,Head
from .chunk_eval import parse_chunk
from .temporal_scene import encode

class Config(BaseModel):
    model_config=ConfigDict(extra='forbid')
    model_path:str
    train_seed:int=21000
    validation_seed:int=31000
    test_seed:int=41000
    train_scenes_per_level:int=Field(default=32,ge=4,le=256)
    validation_scenes_per_level:int=Field(default=8,ge=1)
    test_scenes_per_level:int=Field(default=32,ge=4,le=64)
    states_per_expression:int=Field(default=4,ge=2)
    feature_dimensions:int=Field(default=64,ge=64,le=64)
    epochs:int=Field(default=400,ge=1)
    learning_rate:float=Field(default=.001,gt=0)
    batch_size:int=Field(default=128,ge=1)
    random_seed:int=123
    horizon:int=Field(default=8,ge=8,le=8)
    execute:int=Field(default=4,ge=4,le=4)
    max_steps:int=Field(default=16,ge=16,le=16)

    @model_validator(mode='after')
    def disjoint(self):
        splits=[]
        for key,levels in (('train',3),('validation',3),('test',4)):
            start=getattr(self,key+'_seed'); n=getattr(self,key+'_scenes_per_level')
            splits.append({start+li*1000+i for li in range(levels) for i in range(n)})
        if any(splits[i]&splits[j] for i in range(3) for j in range(i+1,3)): raise ValueError('Scene splits overlap')
        return self

def atomic_json(path,data):
    temp=path.with_suffix('.tmp'); temp.write_text(json.dumps(data,indent=2)); temp.replace(path)

def build_split(config,split):
    levels=('L1','L2','L3','L4') if split=='test' else ('L1','L2','L3')
    cases=[]; attempts=[]
    for li,level in enumerate(levels):
        for i in range(getattr(config,split+'_scenes_per_level')):
            pair,n=make_scene(getattr(config,split+'_seed')+li*1000+i,level,i)
            cases.extend(pair); attempts.append(n)
    return cases,attempts

def observed_features(image,target=None):
    objects,gripper=detect(image)
    base=np.array([2*gripper[0]-1,2*gripper[1]-1,0,0,0,0,1,0],dtype=float)
    xy=np.zeros(64)
    for i,obj in enumerate(objects): xy[2*i:2*i+2]=[2*obj.x-1,2*obj.y-1]
    if target is None: return objects,gripper,base,xy
    extras=np.zeros(64); extras[:2]=[2*target[0]-1,2*target[1]-1]
    extras[2:10]=xy[:8]
    return objects,gripper,base,extras

class Extractor:
    def __init__(self,c,folder):
        import mlx.core as mx
        from mlx_vlm import load
        self.mx=mx; self.c=c; self.folder=folder/'features'; self.folder.mkdir(exist_ok=True)
        self.latencies=[]; self.hits=0
        print('Loading frozen Qwen for referring-expression features',flush=True)
        self.model,self.processor=load(c.model_path); self.model.freeze()
        self.model_config=json.loads((Path(c.model_path)/'config.json').read_text())

    def features(self,image,text):
        from PIL import Image
        from mlx_vlm.utils import prepare_inputs
        from mlx_vlm.prompt_utils import apply_chat_template
        # Scene grounding and current gripper state are separate inputs to the same
        # MLP. Remove the known black marker using observed pixels, not simulator
        # object coordinates. Static scene features can then be reused safely.
        rgb=np.asarray(image).copy(); rgb[np.all(rgb==(15,20,25),axis=2)]=(235,233,222)
        image=Image.fromarray(rgb)
        prompt=apply_chat_template(self.processor,self.model_config,
            'Overhead workspace. Behind means higher in the image; in front of means lower. '
            'Identify the object referred to by this instruction and its spatial relationships '
            'to the other objects. Instruction: '+text,
            num_images=1,enable_thinking=False)
        key=hashlib.sha256((self.c.model_path+'referring-v1'+encode(image)+prompt).encode()).hexdigest()
        path=self.folder/(key+'.npy')
        if path.exists(): self.hits+=1; return np.load(path)
        start=time.perf_counter(); mx=self.mx
        inputs=prepare_inputs(self.processor,images=[image],prompts=prompt,
            image_token_index=self.model_config.get('image_token_index'),min_pixels=50176,max_pixels=50176)
        kwargs={k:v for k,v in inputs.items() if k not in ('input_ids','pixel_values','attention_mask')}
        embeddings=self.model.get_input_embeddings(inputs['input_ids'],inputs.get('pixel_values'),mask=inputs.get('attention_mask'),**kwargs)
        hidden=self.model.language_model.model(inputs['input_ids'],inputs_embeds=embeddings.inputs_embeds,position_ids=embeddings.position_ids)
        vector=hidden[0,-1].astype(mx.float32); mx.eval(vector); result=np.asarray(vector).copy()
        if result.shape!=(4096,) or not np.isfinite(result).all(): raise ValueError('Invalid Qwen features')
        self.latencies.append(time.perf_counter()-start); np.save(path,result); mx.clear_cache()
        if len(self.latencies)%25==0:
            print(json.dumps({'new_feature_pairs':len(self.latencies),'cache_hits':self.hits,'last_seconds':self.latencies[-1]}),flush=True)
        return result

def task_for(data,c): return GroundingTask(data,GroundingConfig(max_steps=c.max_steps))

def demonstration_states(task,c,rng):
    states=[(.5,.5)]; state=GroundingState()
    # Expert rollout supplies on-trajectory states without exposing target identity to the policy.
    for _ in range(c.max_steps):
        action=expert(task,state.gripper,1)[0]
        done=task.advance(state,action)
        if done: break
        if state.step in (3,6): states.append(state.gripper)
    while len(states)<c.states_per_expression:
        p=tuple(rng.uniform(.10,.90,2))
        if all(math.dist(p,(o.x,o.y))>o.radius*math.sqrt(2)+.04 for o in task.data.objects): states.append(p)
    return states[:c.states_per_expression]

def dataset(c,cases,extractor,folder,split):
    path=folder/(split+'_features.npz')
    if path.exists():
        print(f'Using cached {split} matrix',flush=True)
        with np.load(path) as archive: return {k:archive[k] for k in archive.files}
    rows=defaultdict(list); rng=np.random.default_rng(c.random_seed+(0 if split=='train' else 1))
    for ci,data in enumerate(cases):
        task=task_for(data,c); target=next(o for o in data.objects if o.id==data.target_id)
        for point in demonstration_states(task,c,rng):
            image=frame(data.objects,point); _,_,base,coords=observed_features(image)
            _,_,_,oracle=observed_features(image,(target.x,target.y))
            actions=expert(task,point,c.horizon)/task.config.max_delta
            rows['base'].append(base); rows['coordinates'].append(coords); rows['oracle'].append(oracle)
            rows['qwen'].append(extractor.features(image,data.expression.text)); rows['target'].append(actions.flatten())
            rows['seed'].append(data.seed); rows['level'].append(data.expression.level)
        if (ci+1)%16==0: print(f'{split}: {ci+1}/{len(cases)} expressions',flush=True)
    arrays={k:np.asarray(v) for k,v in rows.items()}; np.savez_compressed(path,**arrays); return arrays

def matrix(rows,kind,projection,mean=None,scale=None):
    raw=rows['qwen']@projection if kind=='qwen' else rows[kind]
    if mean is None: mean=raw.mean(0); scale=np.maximum(raw.std(0),.05)
    extras=np.clip((raw-mean)/scale,-6,6)/3
    return np.concatenate((rows['base'],extras),axis=1),mean,scale

def train(c,rows,val,folder):
    rng=np.random.default_rng(c.random_seed)
    # PCA on standardized training features keeps the highest-variance directions; the old random
    # sign projection discarded most of the target-relevant signal (ridge val MSE 0.159 vs 0.064 full).
    feats=rows['qwen']; fmean=feats.mean(0); fscale=feats.std(0)+1e-6
    _,_,vt=np.linalg.svd((feats-fmean)/fscale,full_matrices=False)
    projection=vt[:64].T/fscale[:,None]
    np.save(folder/'projection.npy',projection)
    variants={}
    head_config=TrainConfig(feature_width=72,epochs=c.epochs,batch_size=c.batch_size,
                            learning_rate=c.learning_rate,random_seed=c.random_seed)
    for kind in ('qwen','coordinates','oracle'):
        x,mean,scale=matrix(rows,kind,projection); vx,_,_=matrix(val,kind,projection,mean,scale)
        head=Head(head_config); best=float('inf'); best_epoch=0; curve=[]
        order_rng=np.random.default_rng(c.random_seed); path=folder/(kind+'_head.npz')
        for epoch in range(c.epochs+1):
            if epoch:
                order=order_rng.permutation(len(x))
                for begin in range(0,len(x),c.batch_size):
                    batch=order[begin:begin+c.batch_size]; head.train_batch(x[batch],rows['target'][batch])
            pred,_=head.forward(vx); loss=float(np.mean((pred-val['target'])**2))
            if loss<best: best=loss; best_epoch=epoch; head.save(path)
            if epoch%20==0: curve.append({'epoch':epoch,'validation_mse':loss})
        head.load(path); np.savez_compressed(folder/(kind+'_normalization.npz'),mean=mean,scale=scale)
        variants[kind]={'head':head,'mean':mean,'scale':scale,'projection':projection,'best_epoch':best_epoch,'best_validation_mse':best,'curve':curve}
        print(f'Trained {kind}: validation MSE {best:.5f}, epoch {best_epoch}',flush=True)
    return variants

def inference(kind,variant,image,text,extractor,target=None):
    objects,gripper,base,coords=observed_features(image)
    selected=None; parse_error=None
    if kind=='qwen': raw=extractor.features(image,text)@variant['projection']
    elif kind=='coordinates': raw=coords
    else:
        if kind=='engineered':
            try:
                matches=resolve(objects,parse(text))
                if len(matches)!=1: raise ValueError(f'Parser found {len(matches)} candidates')
                chosen=next(o for o in objects if o.id==matches[0]); target=(chosen.x,chosen.y); selected=target
            except ValueError as exc: parse_error=str(exc)
        if target is None: return [[0.,0.]]*8,selected,parse_error,objects,gripper
        _,_,_,raw=observed_features(image,target)
    extra=np.clip((raw-variant['mean'])/variant['scale'],-6,6)/3
    actions=variant['head'].actions(np.concatenate((base,extra)))
    return actions,selected,parse_error,objects,gripper

def nearest_id(data,point): return min(data.objects,key=lambda o:math.dist(point,(o.x,o.y))).id

def rollout(c,data,kind,variant,extractor,aim=None,label=None):
    task=task_for(data,c); state=GroundingState(); chunks=[]; parser_id=None; initial_plan_id=None
    target=next(o for o in data.objects if o.id==data.target_id)
    while state.step<c.max_steps and state.reached_object_id is None:
        image=task.observe(state); start=time.perf_counter()
        actions,selected,error,detected,observed_gripper=inference(kind,variant,image,data.expression.text,extractor,
                                                                (aim or (target.x,target.y)) if kind=='oracle' else None)
        valid=True
        try: actions=parse_chunk(json.dumps({'actions':actions}),c.horizon,task.config.max_delta)
        except (ValueError,TypeError) as exc: valid=False; error=str(exc); actions=[[0.,0.]]*8
        latency=time.perf_counter()-start; state.chunk_attempts+=1; state.valid_chunks+=int(valid)
        planned=np.asarray(observed_gripper)
        for action in actions: planned=np.clip(planned+action,.06,.94)
        plan_id=nearest_id(data,planned)
        if initial_plan_id is None: initial_plan_id=plan_id
        if selected is not None and parser_id is None: parser_id=nearest_id(data,selected)
        chunk={'step_before':state.step,'gripper_before':state.gripper,'valid':valid,'error':error,
               'predicted_actions':actions,'planned_endpoint':planned.tolist(),'planned_object_id_proxy':plan_id,
               'parser_selected_object_id':nearest_id(data,selected) if selected is not None else None,
               'detected_objects':[o.model_dump() for o in detected], 'latency_seconds':latency,'executed_actions':[]}
        for action in actions[:c.execute]:
            done=task.advance(state,action); chunk['executed_actions'].append(action)
            if done: break
        chunk.update(step_after=state.step,gripper_after=state.gripper,reached_object_id=state.reached_object_id)
        chunks.append(chunk)
    reached=state.reached_object_id
    return {'variant':label or kind,'seed':data.seed,'level':data.expression.level,'family':data.expression.family,
            'expression':data.expression.text,'target_object_id':data.target_id,'reached_object_id':reached,
            'success':reached==data.target_id,'decoy_touch':reached is not None and reached!=data.target_id,
            'timeout':reached is None,'initial_planned_object_id_proxy':initial_plan_id,
            'initial_plan_correct_proxy':initial_plan_id==data.target_id,'parser_selected_object_id':parser_id,
            'parser_selection_correct':parser_id==data.target_id if kind=='engineered' else None,
            'steps':state.step,'endpoint_error_pixels':math.dist(state.gripper,(target.x,target.y))*256,'chunks':chunks}

def summarize(rows):
    result={}
    for kind in ('qwen','coordinates','oracle','engineered','qwen_point'):
        result[kind]={}
        for level in ('L1','L2','L3','L4'):
            part=[r for r in rows if r['variant']==kind and r['level']==level]
            if not part: continue
            chunks=[ch for row in part for ch in row['chunks']]; n=len(part)
            touches=sum(not r['timeout'] for r in part); plans=[r for r in part if r['initial_plan_correct_proxy']]
            result[kind][level]={'episodes':n,'unique_scenes':len({r['seed'] for r in part}),
                'successes':sum(r['success'] for r in part),'success_rate':sum(r['success'] for r in part)/n,
                'decoy_touches':sum(r['decoy_touch'] for r in part),'timeouts':sum(r['timeout'] for r in part),
                'correct_object_fraction_among_touches':sum(r['success'] for r in part)/touches if touches else None,
                'initial_plan_accuracy_proxy':len(plans)/n,
                'success_given_correct_initial_plan_proxy':sum(r['success'] for r in plans)/len(plans) if plans else None,
                'parser_selection_accuracy':sum(r['parser_selection_correct'] for r in part)/n if kind=='engineered' else None,
                'valid_chunk_rate':sum(ch['valid'] for ch in chunks)/len(chunks),
                'mean_endpoint_error_pixels':float(np.mean([r['endpoint_error_pixels'] for r in part])),
                'mean_calls':len(chunks)/n,'mean_pipeline_seconds_including_cache':float(np.mean([ch['latency_seconds'] for ch in chunks]))}
    return result

def counterfactuals(rows):
    result={}
    groups=defaultdict(list)
    for row in rows: groups[(row['variant'],row['level'],row['seed'])].append(row)
    for kind in ('qwen','coordinates','oracle','engineered','qwen_point'):
        result[kind]={}
        for level in ('L1','L2','L3','L4'):
            pairs=[p for (k,l,_),p in groups.items() if k==kind and l==level and len(p)==2]
            if pairs:
                result[kind][level]={'pairs':len(pairs),'both_instructions_succeeded':sum(all(r['success'] for r in p) for p in pairs),
                                     'reached_different_objects':sum(p[0]['reached_object_id'] is not None and p[1]['reached_object_id'] is not None and p[0]['reached_object_id']!=p[1]['reached_object_id'] for p in pairs)}
    return result

def validate_benchmark(c,folder):
    folder.mkdir(parents=True,exist_ok=True); all_cases={}; stats={}; parser_checks=[]; detection_checks=[]
    for split in ('train','validation','test'):
        cases,attempts=build_split(c,split); all_cases[split]=cases
        atomic_json(folder/(split+'_scenes.json'),[d.model_dump(mode='json') for d in cases])
        level_stats={}
        for level in sorted({d.expression.level for d in cases}):
            data=[d for d in cases if d.expression.level==level]; targets=[next(o for o in d.objects if o.id==d.target_id) for d in data]
            level_stats[level]={'scenes':len({d.seed for d in data}),'expressions':len(data),
                'object_counts':{str(n):sum(len(d.objects)==n for d in data) for n in (2,3,4)},
                'target_mean_xy':[float(np.mean([getattr(o,k) for o in targets])) for k in ('x','y')],
                'target_quadrants':{f'{x},{y}':sum((o.x>.5)==x and (o.y>.5)==y for o in targets) for x in (False,True) for y in (False,True)},
                'families':{f:sum(d.expression.family==f for d in data) for f in sorted({d.expression.family for d in data})}}
        stats[split]={'levels':level_stats,'mean_scene_generation_attempts':float(np.mean(attempts)),'max_attempts':max(attempts)}
        for ci,d in enumerate(cases):
            parsed=parse(d.expression.text)
            parser_checks.append(resolve(d.objects,parsed)==[d.target_id])
            objects,_=detect(frame(d.objects,(.5,.5))); selected=resolve(objects,parsed)
            detection_checks.append(len(selected)==1 and nearest_id(d,(objects[selected[0]].x,objects[selected[0]].y))==d.target_id)
            task=task_for(d,c); state=GroundingState()
            while state.step<c.max_steps and state.reached_object_id is None:
                if task.advance(state,expert(task,state.gripper,1)[0]): break
            if state.reached_object_id!=d.target_id: raise ValueError(f'Expert failed {d.seed}: {d.expression.text}')
            if (ci+1)%32==0: print(f'Audited {split}: {ci+1}/{len(cases)} expressions',flush=True)
    if not all(parser_checks): raise ValueError('Text parser disagrees with semantic generator')
    stats['text_parser_gt_semantic_agreement']=sum(parser_checks)/len(parser_checks)
    stats['image_detector_parser_selection_accuracy']=sum(detection_checks)/len(detection_checks)
    stats['uniqueness_verified_expressions']=len(parser_checks)
    import inspect
    stats['engineering_scope']={'parser_lines':len(inspect.getsource(parse).splitlines()),
                               'semantic_resolver_lines':len(inspect.getsource(resolve).splitlines()),
                               'image_detector_lines':len(inspect.getsource(detect).splitlines()),
                               'note':'Full fixed grammar supports L1–L4. Engineering cost does not necessarily grow steeply for these finite templates.'}
    stats['counterfactual_design']='Two different uniquely specified targets per identical scene; target is not marked. Any deterministic no-text policy has <=50% success on each complete paired split.'
    stats['behind_definition']='higher image y-direction: target y < anchor y - 0.06 (top-down screen coordinates)'
    stats['between_definition']='projection within (0.15,0.85) of anchor segment, perpendicular distance <0.10 normalized workspace'
    atomic_json(folder/'data_audit.json',stats)
    return all_cases,stats

def report(folder,report):
    lines=['# Referring-expression grounding pilot','',
           'Frozen 4-bit Qwen3.5-9B; last prompt-token decoder state projected to 64 dimensions. All learned heads: 72 → 128 → 128 → 16, tanh; H=8, r=4.', '',
           'Qwen receives the observed scene with the known black gripper marker removed by pixel color. Current gripper coordinates are detected from the original image and supplied separately to the same head. Static scene embeddings are cached; no simulator target coordinates enter the Qwen policy.', '',
           'Training and validation: L1–L3 only. L4 between/not-red excluded from demonstrations and checkpoint selection. Full engineered parser explicitly supports all four levels; its L4 result is not zero-shot learning.','',
           '| Level | Qwen | Coordinates only | Oracle selection | Engineered parser |','|---|---:|---:|---:|---:|']
    for level in ('L1','L2','L3','L4'):
        values=[]
        for kind in ('qwen','coordinates','oracle','engineered','qwen_point'):
            m=report['summaries'][kind][level]; values.append(f"{m['successes']}/{m['episodes']}")
        lines.append('| '+level+' | '+' | '.join(values)+' |')
    lines.extend(['','## Interpretation limits','',
        '- Actual reached-object ID, decoy contacts, and timeouts are logged separately. Oracle-selected control measures the reaching bottleneck.',
        '- The first predicted chunk endpoint nearest-object metric is a selection proxy, not a separate learned classifier.',
        '- Each scene supplies two instructions for different targets. A deterministic coordinates-only policy cannot exceed 50% paired success.',
        '- Episodes sharing a scene are correlated; report unique scene counts, not twice as many independent scenes.',
        '- One optimizer seed, small synthetic data, static objects, fixed template grammar, and a 64-dimensional projected representation. No cloth dynamics or physical robot claim.',
        '- Qwen pretraining may contain between/negation concepts: L4 is held out from our action-head training, not from all pretraining.',
        '- Simulator pauses during inference. Cached rollout latency is not a real-time measurement; fresh extraction is reported separately.','',
        '## Fresh Qwen extraction latency','',json.dumps(report.get('feature_latency',{}),indent=2)])
    (folder/'report.md').write_text('\n'.join(lines)+'\n')

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',type=Path,default=Path(__file__).with_name('referring.toml'))
    parser.add_argument('--output',type=Path,default=Path('runs/referring')); parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args(); c=Config.model_validate(tomllib.loads(args.config.read_text())); folder=args.output
    fingerprint=hashlib.sha256((args.config.read_text()+Path(__file__).read_text()+Path(__file__).with_name('referring_scene.py').read_text()).encode()).hexdigest()
    old_manifest=folder/'manifest.json'
    if old_manifest.exists() and json.loads(old_manifest.read_text())['fingerprint']!=fingerprint:
        raise ValueError('Config/source changed: choose a fresh output directory to avoid stale matrices')
    cases,audit=validate_benchmark(c,folder)
    atomic_json(old_manifest,{'fingerprint':fingerprint,'config':c.model_dump(),'train_levels':['L1','L2','L3'],'held_out':['between','not_red'],
                             'parser_scope':'Full explicit template grammar L1–L4; no forced degradation','head_parameters':27920})
    print(json.dumps({'audit':audit},indent=2),flush=True)
    if args.validate_only: return
    extractor=Extractor(c,folder); rows=dataset(c,cases['train'],extractor,folder,'train'); val=dataset(c,cases['validation'],extractor,folder,'validation')
    variants=train(c,rows,val,folder)
    output={'completed':False,'config':c.model_dump(),'audit':audit,'training_samples':len(rows['target']),'validation_samples':len(val['target']),
            'learning_curves':{k:{f:v[f] for f in ('best_epoch','best_validation_mse','curve')} for k,v in variants.items()}}
    atomic_json(folder/'comparison.json',output)
    traces_path=folder/'rollouts.jsonl'; existing=[]; completed=set()
    if traces_path.exists():
        existing=[json.loads(line) for line in traces_path.read_text().splitlines() if line]
        completed={(r['variant'],r['seed'],r['expression']) for r in existing}
    with traces_path.open('a') as stream:
        for kind in ('coordinates','oracle','engineered','qwen'):
            variant=variants['oracle' if kind=='engineered' else kind]
            for ci,data in enumerate(cases['test']):
                key=(kind,data.seed,data.expression.text)
                if key in completed: continue
                row=rollout(c,data,kind,variant,extractor); existing.append(row)
                stream.write(json.dumps(row)+'\n'); stream.flush()
                if (ci+1)%16==0: print(f'Evaluated {kind}: {ci+1}/{len(cases["test"])}',flush=True)
    l=np.asarray(extractor.latencies)
    by_family={}
    for family in sorted({r['family'] for r in existing}):
        by_family[family]=summarize([r for r in existing if r['family']==family])
    four_objects=summarize([r for r in existing if len(next(d.objects for d in cases['test'] if d.seed==r['seed']))==4])
    output.update(completed=True,summaries=summarize(existing),family_summaries=by_family,four_object_summaries=four_objects,
                  counterfactual_pairs=counterfactuals(existing),
                  feature_latency={'fresh_calls':len(l),'cache_hits':extractor.hits,
                    'median_seconds':float(np.median(l)) if len(l) else None,
                    'p95_seconds':float(np.percentile(l,95)) if len(l) else None,
                    'p99_seconds':float(np.percentile(l,99)) if len(l) else None,
                    'peak_memory_gb':extractor.mx.get_peak_memory()/1e9})
    atomic_json(folder/'comparison.json',output); report(folder,output); print(json.dumps(output['summaries'],indent=2),flush=True)

if __name__=='__main__': main()
