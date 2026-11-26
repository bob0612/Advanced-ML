import pickle, sys, json
sys.path.insert(0,'/home/user/mp3/dev'); sys.path.insert(0,'/home/user/mp3/submission')
from oracle import Split, save
from mp3agent import sqlops
recs = pickle.load(open(sys.argv[1],'rb'))
EDITS = {'lower':lambda s:[sqlops.lower_literals(s)],'title':lambda s:[sqlops.title_literals(s)],'like':lambda s:[sqlops.like_literals(s)],
 'ineq':lambda s:[sqlops.flip_inequalities(s)],'order':lambda s:[sqlops.flip_order(s)],'nolimit':lambda s:[sqlops.drop_limit(s)],
 'limit1':lambda s:[sqlops.add_limit_one(s)],'dropconj':sqlops.drop_conjuncts,
 'align':lambda s:[], }
for split in ['train','validation']:
    S=Split(split); stats={k:[0,0,0] for k in EDITS}; n_trig=0; gain=set()
    def f(task):
        row=task['row']; pool=task['pool']
        if row['example']['row_count']==0 or any(p['feedback']['matches_example'] for p in pool): return None
        res={}
        for k,fn in EDITS.items():
            out=[]
            for p in pool:
                for s in fn(p['sql']) or []:
                    if s:
                        s2 = sqlops.align_projection(s,row['example']['columns']) or s
                        out.append((S.matches(row,s2), S.correct(row,s2)))
            res[k]=out
        return res
    res=S.pmap(f,recs[split])
    for t,r in zip(recs[split],res):
        if r is None: continue
        n_trig+=1
        for k,out in r.items():
            stats[k][0]+=len(out); stats[k][1]+=sum(m and c for m,c in out)>0; stats[k][2]+=sum(m and not c for m,c in out)>0
            if any(m and c for m,c in out): gain.add(t['row']['id'])
    print(split,'triggered',n_trig,'tasks gaining a matching correct cand',len(gain))
    for k,v in stats.items(): print('  ',k,'ncand',v[0],'tasks w/ match&correct',v[1],'tasks w/ match&wrong',v[2])
    save()
