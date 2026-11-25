"""Grouped (by db) 5-fold CV on train: per-task selection accuracy with all candidates checked."""
import json, sys, pickle, numpy as np
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,'/home/user/mp3/dev'); sys.path.insert(0,'/home/user/mp3/submission')
from oracle import Split, save
from mp3agent import features, sqlops
from mp3agent.candidates import build_pool
from sklearn.linear_model import LogisticRegression
hints=json.load(open('/home/user/mp3/pkg/MP3_SQL_student_package/mp3_sql/data/repair_hints.json'))
rw=tuple(sys.argv[1].split(','))
import os
MODEL=os.environ.get('MODEL','lr')
S=Split('train')
def task(row):
    obs={'task':{'id':row['id'],'question':row['question']},'issue_sql':row['issue_sql'],'example':row['example'],'ordered':row['ordered']}
    pool=build_pool(row['issue_sql'],hints[row['id']]['suggestions'],obs,rw)
    fbs=[];lab=[]
    for c in pool:
        r=S.public_result(row,c['sql'])
        fbs.append({"ok":r['ok'],"columns":r['columns'],"rows":r['rows'][:8],"row_count":len(r['rows']),"truncated":len(r['rows'])>8,"error":r['error'],"matches_example":S.matches(row,c['sql'])})
        lab.append(int(S.correct(row,c['sql'])))
    checked={c['sql']:f for c,f in zip(pool,fbs)}
    ctx={'draft_feedback':fbs[0],'draft_sql':row['issue_sql'],'max_votes':max(c['votes'] for c in pool),'n_hints':max(1,sum(c['votes'] for c in pool)),
         'draft_struct':sqlops.features_of(row['issue_sql']),'twins':features.twin_matches(pool,checked)}
    return [features.vector(obs,c,f,ctx) for c,f in zip(pool,fbs)], lab
with ThreadPoolExecutor(8) as ex: res=list(ex.map(task,S.rows))
save()
dbs=sorted({r['db_id'] for r in S.rows}); rng=np.random.RandomState(0); perm=rng.permutation(len(dbs)); fold={dbs[i]:k%5 for k,i in enumerate(perm)}
tot=0; orc=sum(max(l) for _,l in res)
for C in [float(x) for x in (sys.argv[2].split(',') if len(sys.argv)>2 else ['1'])]:
    tot=0
    for k in range(5):
        tr=[i for i,r in enumerate(S.rows) if fold[r['db_id']]!=k]; te=[i for i,r in enumerate(S.rows) if fold[r['db_id']]==k]
        X=np.array([x for i in tr for x in res[i][0]]); y=np.array([v for i in tr for v in res[i][1]])
        if MODEL=='lr': m=LogisticRegression(C=C,max_iter=20000).fit(X,y); f=m.decision_function
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier
            m=HistGradientBoostingClassifier(max_depth=int(C),learning_rate=0.05,max_iter=200,random_state=0).fit(X,y); f=lambda Z:m.predict_proba(Z)[:,1]
        for i in te:
            s=f(np.array(res[i][0])); tot+=res[i][1][int(np.argmax(s))]
    print(','.join(rw),'C',C,'CV',tot,'/480 oracle',orc)
