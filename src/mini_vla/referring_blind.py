"""Blind-parser tracker (parser written from the task description only) + the unchanged coordinate head, on all 256 test episodes."""
from __future__ import annotations
import argparse, json, sys, tomllib, collections
from pathlib import Path
import numpy as np
from . import referring_experiment as ex
from .referring_experiment import Config, build_split, rollout, atomic_json
from .referring_paraphrase import paraphrase
from .trained_chunk import TrainConfig, Head

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',type=Path,required=True); ap.add_argument('--heads',type=Path,required=True)
    ap.add_argument('--parser_dir',type=Path,required=True); ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); c=Config.model_validate(tomllib.loads(a.config.read_text())); a.output.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(a.parser_dir)); import blind_parser
    ex.parse=lambda text: text
    def resolve(objects,text):
        i=blind_parser.select(text,[{'id':o.id,'color':o.color,'shape':o.shape,'size':o.size,'x':o.x,'y':o.y} for o in objects])
        return [] if i is None else [i]
    ex.resolve=resolve
    head=Head(TrainConfig(feature_width=72,epochs=1,batch_size=128,learning_rate=.001,random_seed=c.random_seed)); head.load(a.heads/'oracle_head.npz')
    norm=np.load(a.heads/'oracle_normalization.npz'); variant={'head':head,'mean':norm['mean'],'scale':norm['scale']}
    cases,_=build_split(c,'test'); out={}
    for name,wording in (('original',lambda d,i:d.expression.text),('paraphrased',lambda d,i:paraphrase(d.expression,i))):
        lv=collections.defaultdict(lambda:[0,0,0,0,0]); rows=[]
        for i,data in enumerate(cases):
            text=wording(data,i)
            d=data.model_copy(update={'expression':data.expression.model_copy(update={'text':text})})
            r=rollout(c,d,'engineered',variant,None,label='blind_tracker'); r['text']=text; rows.append(r)
            unresolved=r['parser_selected_object_id'] is None
            s=lv[r['level']]; s[0]+=1; s[1]+=r['success']; s[2]+=r['decoy_touch']; s[3]+=r['timeout']; s[4]+=unresolved
        (a.output/f'{name}.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
        out[name]=dict(sorted(lv.items())); print(name,'[episodes,successes,decoy,timeout,parser_returned_none]',out[name],flush=True)
    atomic_json(a.output/'comparison.json',out)
if __name__=='__main__': main()
