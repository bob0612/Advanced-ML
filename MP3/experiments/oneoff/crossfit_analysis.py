"""One-off analysis on the pod (train only): LLM gains/losses vs. no-LLM on cross-fit folds."""
import json,glob,collections
C=collections.Counter(); ex=[]
for k in range(5):
    D=f'/workspace/mp3/runs/cf/f{k}'
    A={json.loads(l)['id']:json.loads(l) for l in open(D+'/run.jsonl')}
    B={json.loads(l)['id']:json.loads(l) for l in open(D+'/nollm.jsonl')}
    for i,a in A.items():
        b=B.get(i)
        if b is None: continue
        C['n']+=1; C['llm_pass']+=a['passed']; C['nollm_pass']+=b['passed']
        called='schema' in a['tools']
        C['called']+=called
        # tag llm checks
        seen=False; tags=[]
        for t in a['tools']:
            if t=='schema': seen=True
            if t=='check': tags.append(seen)
        llmc=[c for c,t in zip(a['checks'],tags) if t]; poolc=[c for c,t in zip(a['checks'],tags) if not t]
        if a['passed']!=b['passed']: C['gain' if a['passed'] else 'loss']+=1
        if not a['passed']:
            anyp=any(c['correct'] for c in poolc); anyl=any(c['correct'] for c in llmc)
            sub_llm=any(c['sql']==a['sql'] for c in llmc)
            key=f"fail empty={int(a['empty'])} called={int(called)} pool_correct={int(anyp)} llm_correct={int(anyl)} submitted_llm={int(sub_llm)} sub_match={int(any(c['sql']==a['sql'] and c['match'] for c in a['checks']))}"
            C[key]+=1
        if called:
            C['llm_any_correct']+=any(c['correct'] for c in llmc)
            C['llm_any_match']+=any(c['match'] for c in llmc)
            C['llm_match_and_correct']+=any(c['match'] and c['correct'] for c in llmc)
            C['llm_match_wrong_only']+=(any(c['match'] for c in llmc) and not any(c['match'] and c['correct'] for c in llmc))
            C['ncheck_llm']+=len(llmc)
for k,v in sorted(C.items()): print(v,k)
print('--- gains/losses detail')
for k in range(5):
    D=f'/workspace/mp3/runs/cf/f{k}'
    A={json.loads(l)['id']:json.loads(l) for l in open(D+'/run.jsonl')}
    B={json.loads(l)['id']:json.loads(l) for l in open(D+'/nollm.jsonl')}
    for i,b in B.items():
        a=A[i]
        if a['passed']==b['passed']: continue
        bm=any(c['sql']==b['sql'] and c['match'] for c in b['checks'])
        print('GAIN' if a['passed'] else 'LOSS','empty',int(a['empty']),'nollm_sub_match',int(bm),
              'nollm_sub_checked_in_llmrun',int(b['sql'] in [c['sql'] for c in a['checks']]))
