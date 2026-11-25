"""Audit complete or partial behavioral checkpoints, without inference or scoring."""
import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from statistics import mean

CONDITIONS=('baseline','s1','s2','s1_then_s2','s2_then_s1','joint')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('run',type=Path)
    p.add_argument('source',type=Path)
    p.add_argument('output',type=Path)
    args=p.parse_args()
    result={'scope':'checkpoint inspection; no independent blinded human ratings',
            'run':str(args.run),'conditions':{},'validity_issues':[]}
    settings=args.run/'settings.json'
    if not settings.exists():
        result['validity_issues'].append('No generation settings/checkpoints: generation did not start')
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        return
    config=json.loads(settings.read_text())
    result['settings']=config
    dataset=args.source/'datasets/4.1_self_other_420_scenarios.json'
    assert hashlib.sha256(dataset.read_bytes()).hexdigest()==config['dataset_sha256']
    original_prompts={r['id']:r for r in json.loads(dataset.read_text())}
    expected=set(config['prompt_ids'])
    tokens=args.run/'prompt_token_ids.json'
    token_records=json.loads(tokens.read_text())
    assert set(token_records)==expected
    result['input_tokens_sha256']=hashlib.sha256(tokens.read_bytes()).hexdigest()
    published=json.loads((args.source/'results/appC_ablation/by_prompt_Gemma_2_2B_instruct.json').read_text())
    result['historical_generation_sha256']=hashlib.sha256(
        (args.source/'results/appC_ablation/by_prompt_Gemma_2_2B_instruct.json').read_bytes()).hexdigest()
    historical_names={'baseline':'baseline','s1':'s1','s2':'s2','s1_then_s2':'s1s2'}
    all_rows={}
    for condition in CONDITIONS:
        rows=[]
        seconds=0
        for path in sorted(args.run.glob(f'{condition}.[0-9][0-9][0-9].json')):
            batch=json.loads(path.read_text())
            assert batch['condition']==condition
            seconds+=batch['seconds']
            rows.extend(batch['rows'])
        ids=[r['id'] for r in rows]
        assert len(ids)==len(set(ids)) and set(ids)<=expected
        for row in rows:
            assert len(row['generated_ids_with_batch_padding'])<=60
            assert row['prompt_sha256']==hashlib.sha256(original_prompts[row['id']]['text'].encode()).hexdigest()
            assert all(math.isfinite(x) for x in row['fixed_avg_abs_projection_with_batch_padding'])
        all_rows[condition]={r['id']:r for r in rows}
        report={'completed_prompts':len(rows),'missing_ids':sorted(expected-set(ids)),
                'batch_seconds':seconds,'category_counts':dict(Counter(r['category'] for r in rows)),
                'empty_output_ids':[r['id'] for r in rows if not r['generation'].strip()]}
        if report['missing_ids']:
            result['validity_issues'].append(f'{condition}: incomplete checkpoint coverage')
        if condition in historical_names:
            label=historical_names[condition]
            matches=[]
            for row in rows:
                old=published[str(row['id'])][label]
                matches.append({'id':row['id'],'exact_match':old==row['generation'],
                                'whitespace_match':old.strip()==row['generation'].strip()})
            report['historical_replay']={'n':len(matches),
                'exact_matches':sum(r['exact_match'] for r in matches),
                'whitespace_matches':sum(r['whitespace_match'] for r in matches),
                'mismatch_ids':[r['id'] for r in matches if not r['exact_match']]}
        writers=args.run/f'{condition}.writers.json'
        if writers.exists():
            audit=json.loads(writers.read_text())
            assert all(math.isfinite(x) for v in audit.values() for x in v['relative_to_original'])
            report['writers']={'n':len(audit),
                'max_relative_leakage_each_direction':[max(v['relative_to_original'][i] for v in audit.values()) for i in range(2)],
                'mean_relative_writer_change_unweighted':mean(v['relative_writer_change'] for v in audit.values()),
                'max_relative_writer_change':max(v['relative_writer_change'] for v in audit.values())}
        elif rows:
            result['validity_issues'].append(f'{condition}: writer audit missing')
        result['conditions'][condition]=report
    result['complete_six_condition_panel']=all(set(rows)==expected for rows in all_rows.values())
    result['paired_text_changes']={}
    for before,after in [('baseline','s1_then_s2'),('s1_then_s2','s2_then_s1'),('s1_then_s2','joint')]:
        common=set(all_rows[before])&set(all_rows[after])
        result['paired_text_changes'][f'{before} -> {after}']={
            'paired_n':len(common),'exact_text_changed':sum(all_rows[before][i]['generation']!=all_rows[after][i]['generation'] for i in common),
            'interpretation':'Wording change only; does not establish humor, deflection, coherence, relevance or competence.'}
    result['total_batch_seconds']=sum(x['batch_seconds'] for x in result['conditions'].values())
    result['ratings_available']=False
    result['general_competence_control']='Not run; hostile conversations do not establish preserved general competence'
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
