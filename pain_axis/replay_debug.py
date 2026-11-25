"""Frozen historical replay diagnostic; companion rows only restore author batches."""
import argparse
import gc
import json
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pain_axis.geometry import apply_condition, file_sha, operators
from pain_axis.run import MODEL, PROMPTS, read_directions, save_json, select_prompts


def main():
    p=argparse.ArgumentParser()
    p.add_argument('source',type=Path);p.add_argument('pr',type=Path)
    p.add_argument('revision');p.add_argument('vectors',type=Path)
    p.add_argument('smoke',type=Path);p.add_argument('out',type=Path)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(42);torch.backends.cuda.matmul.allow_tf32=False
    rows=select_prompts(args.source/PROMPTS,False)
    targets={r['id'] for r in select_prompts(args.source/PROMPTS,True)}
    groups=[(i//8,rows[i:i+8]) for i in range(0,len(rows),8)
            if any(r['id'] in targets for r in rows[i:i+8])]
    assert len(groups)==5 and sum(map(lambda g:len(g[1]),groups))==40
    smoke_tokens=json.loads((args.smoke/'prompt_token_ids.json').read_text())
    published=json.loads((args.source/'results/appC_ablation/by_prompt_Gemma_2_2B_instruct.json').read_text())
    tok=AutoTokenizer.from_pretrained(MODEL,revision=args.revision)
    if tok.pad_token is None:tok.pad_token=tok.eos_token
    historical,corrected=operators(args.source,args.pr)
    v=read_directions(args.vectors,'regenerated',args.revision,'cuda','bfloat16')
    result={'scope':'10 fixed targets, 30 companion rows restoring original batches; not the 100-prompt six-way panel',
            'runner_sha256':file_sha(Path(__file__)),'revision':args.revision,
            'vectors_sha256':file_sha(args.vectors),'target_ids':sorted(targets),
            'author_batch_indices':[i for i,_ in groups],
            'disable_compile':True,'backends':{}}
    for backend in ['default','eager']:
        opts={} if backend=='default' else {'attn_implementation':'eager'}
        model=AutoModelForCausalLM.from_pretrained(MODEL,revision=args.revision,
                torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,device_map={'':'cuda'},**opts).eval()
        snap=historical['snapshot_weights'](model)
        result['backends'][backend]={'actual_attention':model.config._attn_implementation,'conditions':{}}
        for condition,old_name in [('baseline','baseline'),('s1_then_s2','s1s2')]:
            historical['restore_weights'](model,snap)
            apply_condition(model,historical,corrected,v,condition)
            scored=[]
            for index,batch in groups:
                path=args.out/f'{backend}.{condition}.{index:03d}.json'
                texts=[historical['render_prompt'](r,tok,'chat') for r in batch]
                enc=tok(texts,return_tensors='pt',padding=True,padding_side='left').to('cuda')
                for row,ids,mask in zip(batch,enc.input_ids,enc.attention_mask):
                    if row['id'] in targets:assert ids[mask.bool()].tolist()==smoke_tokens[row['id']]
                torch.cuda.synchronize();start=perf_counter()
                with torch.inference_mode():
                    out=model.generate(**enc,max_new_tokens=60,do_sample=False,
                                       pad_token_id=tok.pad_token_id,disable_compile=True)
                torch.cuda.synchronize();elapsed=perf_counter()-start
                generated=out[:,enc.input_ids.shape[1]:]
                records=[]
                for row,text,ids in zip(batch,tok.batch_decode(generated,skip_special_tokens=True),generated):
                    item={'id':row['id'],'target':row['id'] in targets,'generation':text,
                          'generated_ids_with_batch_padding':ids.tolist()}
                    if item['target']:
                        item['historical_exact_match']=text==published[row['id']][old_name]
                        scored.append(item)
                    records.append(item)
                save_json(path,{'backend':backend,'condition':condition,'author_batch':index,
                                'seconds':elapsed,'rows':records})
                print(backend,condition,index,f'{elapsed:.3f}s',flush=True)
            assert len(scored)==10
            result['backends'][backend]['conditions'][condition]={
                'exact_matches':sum(r['historical_exact_match'] for r in scored),
                'mismatch_ids':[r['id'] for r in scored if not r['historical_exact_match']]}
        del model,snap;gc.collect();torch.cuda.empty_cache()
        save_json(args.out/'summary.json',result)
    # Preregistered gate: the author's default attention and original batch
    # composition must replay both controls exactly before confirmatory scaling.
    result['full_panel_gate_passed']=all(
        c['exact_matches']==10 for c in result['backends']['default']['conditions'].values())
    save_json(args.out/'summary.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
