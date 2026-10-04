"""Execute frozen-Qwen movement chunks in a small kinematic reaching environment."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import re
import time
import tomllib
import urllib.request
from collections import defaultdict
from pathlib import Path
from datetime import datetime,timezone
from PIL import Image
from .chunk_control import ChunkPilotConfig,ChunkReachTaskset,ReachState
from .temporal_scene import decode


def request_chunk(model,base_url,inputs):
    payload={'model':model,'input':inputs,'temperature':0,'reasoning':'off',
             'max_output_tokens':512,'store':False}
    request=urllib.request.Request(base_url.rstrip('/')+'/chat',data=json.dumps(payload).encode(),
                                  headers={'Content-Type':'application/json'})
    if os.environ.get('LM_API_TOKEN'):
        request.add_header('Authorization','Bearer '+os.environ['LM_API_TOKEN'])
    start=time.perf_counter()
    with urllib.request.urlopen(request,timeout=240) as response:
        result=json.load(response)
    replies=[item['content'] for item in result.get('output',[]) if item.get('type')=='message' and isinstance(item.get('content'),str)]
    return replies[-1] if replies else '',time.perf_counter()-start


def parse_chunk(reply,horizon,max_delta):
    # Accept one Markdown code wrapper; still reject invalid or out-of-bounds actions.
    clean=reply.strip()
    fence=re.fullmatch(r'```(?:json)?\s*(.*?)\s*```',clean,re.DOTALL)
    parsed=json.loads(fence.group(1) if fence else clean)
    if not isinstance(parsed,dict) or set(parsed)!={'actions'} or not isinstance(parsed['actions'],list) or len(parsed['actions'])!=horizon:
        raise ValueError('Expected exactly horizon actions')
    actions=[]
    for action in parsed['actions']:
        if not isinstance(action,list) or len(action)!=2 or any(type(v) not in (float,int) for v in action):
            raise ValueError('Each action must be [dx,dy]')
        if any(not math.isfinite(v) or abs(v)>max_delta+1e-9 for v in action):
            raise ValueError('Movement exceeds action bounds')
        actions.append(tuple(float(v) for v in action))
    return actions


def summarize(episodes):
    groups=defaultdict(list)
    for episode in episodes:
        groups[(episode['policy'],episode['condition'])].append(episode)
    return {f'{policy}/{condition}':{
        'n':len(rows),'successes':sum(r['success'] for r in rows),
        'success_rate':sum(r['success'] for r in rows)/len(rows),
        'mean_endpoint_error_pixels':sum(r['endpoint_error_pixels'] for r in rows)/len(rows),
        'mean_executed_steps':sum(r['steps'] for r in rows)/len(rows),
        'mean_policy_calls':sum(len(r['chunks']) for r in rows)/len(rows),
        'invalid_chunk_count':sum(sum(not c['valid'] for c in r['chunks']) for r in rows),
        'valid_chunk_rate':sum(c['valid'] for r in rows for c in r['chunks'])/max(1,sum(len(r['chunks']) for r in rows)),
        'mean_request_seconds':sum(c['latency_seconds'] for r in rows for c in r['chunks'])/max(1,sum(len(r['chunks']) for r in rows)),
    } for (policy,condition),rows in groups.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path(__file__).with_name('chunk_h8_r2.toml'))
    parser.add_argument('--model',default='qwen/qwen3.5-9b')
    parser.add_argument('--base-url',default='http://localhost:1234/api/v1')
    parser.add_argument('--output',type=Path,default=Path('runs/qwen_chunk_h8_r2.json'))
    args=parser.parse_args()
    with args.config.open('rb') as f: config=ChunkPilotConfig.model_validate(tomllib.load(f))
    tasks=ChunkReachTaskset(config).load()
    report={'created_at_utc':datetime.now(timezone.utc).isoformat(),
            'config':config.model_dump(mode='json'),'model':args.model,'base_url':args.base_url,
            'policy_type':'frozen_Qwen_prompted_action_chunks','physics':'2D_kinematic_no_contact',
            'temperature':0,'reasoning':'off','max_output_tokens':512,
            'failure_handling':'invalid chunks execute zero displacement for the prefix; no retries',
            'success_radius_pixels':8.96,'completed':False,'episodes':[]}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    artifact_dir=args.output.with_suffix('')
    artifact_dir.mkdir(parents=True,exist_ok=True)
    for task in tasks:
        # Ground-truth reference checks task feasibility. It is not a learned-policy comparator.
        oracle=ReachState()
        while oracle.step<task.config.max_steps and not oracle.success:
            _,target=task.geometry(oracle.step)
            actions=[]
            x,y=oracle.gripper
            for _ in range(config.horizon):
                dx=max(-config.max_delta,min(config.max_delta,target[0]/256-x))
                dy=max(-config.max_delta,min(config.max_delta,target[1]/256-y))
                actions.append((dx,dy));x+=dx;y+=dy
            for action in actions[:config.execute]:
                if task.advance(oracle,action): break
        report['episodes'].append({'policy':'ground_truth_reference','seed':task.data.seed,
            'condition':'arm_occluded' if task.data.occluded else 'clear_view','success':oracle.success,
            'endpoint_error_pixels':task.error(oracle),'steps':oracle.step,'chunks':[]})
        state=ReachState()
        frames=[decode(task.observe(state))]
        history=None
        chunks=[]
        while state.step<task.config.max_steps and not state.success:
            current=task.observe(state)
            prompt=(f'Control the red-outlined gripper to reach the orange marker attached to a blue cloth corner. '
                f'The camera is fixed; cloth may translate and rotate. A dark arm may hide the marker. '
                f'Current gripper position (proprioception) is x={state.gripper[0]:.4f}, y={state.gripper[1]:.4f}. '
                f'Coordinates increase rightward/downward on a 256x256 image. Predict exactly {config.horizon} '
                f'future incremental movements [dx,dy] in normalized image coordinates. Each component must lie '
                f'between {-config.max_delta} and {config.max_delta}. Moves are applied sequentially to the gripper. '
                f'Only the first {config.execute} moves will execute before a fresh observation. '
                f'Return exactly JSON with one key "actions" containing {config.horizon} pairs, no explanation or markdown. '
                'Use zero movements for remaining steps if you expect to reach the marker. '
                'Do not report absolute positions; report movement increments.')
            inputs=[{'type':'text','content':prompt}]
            if history is not None:
                inputs.extend([{'type':'text','content':'PREVIOUS observation:'},
                               {'type':'image','data_url':history}])
            inputs.extend([{'type':'text','content':'CURRENT observation:'},{'type':'image','data_url':current}])
            start_position=state.gripper
            reply,latency=request_chunk(args.model,args.base_url,inputs)
            valid=True;parse_error=None
            try: actions=parse_chunk(reply,config.horizon,config.max_delta)
            except (ValueError,TypeError,KeyError) as error:
                valid=False;parse_error=str(error);actions=[(0.,0.)]*config.horizon
            state.chunk_attempts+=1
            state.valid_chunks+=int(valid)
            row={'step_before':state.step,'gripper_before':start_position,'reply':reply,
                 'valid':valid,'parse_error':parse_error,'latency_seconds':latency,
                 'markdown_wrapped':reply.strip().startswith('```'),
                 'predicted_actions':actions,'executed_actions':[],
                 'input_sha256':hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()}
            for action in actions[:config.execute]:
                done=task.advance(state,action)
                row['executed_actions'].append(action)
                frames.append(decode(task.observe(state)))
                if done: break
            row.update(step_after=state.step,gripper_after=state.gripper,error_after_pixels=task.error(state))
            chunks.append(row);history=current
            partial={'policy':'qwen','seed':task.data.seed,'condition':'arm_occluded' if task.data.occluded else 'clear_view',
                'success':state.success,'endpoint_error_pixels':task.error(state),'steps':state.step,
                'path_length_pixels':state.path_length_pixels,'chunks':chunks}
            report['current_episode']=partial
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps({'seed':task.data.seed,'condition':partial['condition'],'step':state.step,
                              'valid':valid,'error_pixels':task.error(state),'success':state.success}),flush=True)
        report.pop('current_episode',None)
        report['episodes'].append(partial)
        frames[0].save(artifact_dir/f'{task.data.seed}-{partial["condition"]}.gif',save_all=True,
                       append_images=frames[1:],duration=250,loop=0)
        report['summary']=summarize(report['episodes'])
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    report['completed']=True
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['summary'],indent=2),flush=True)

if __name__=='__main__': main()
