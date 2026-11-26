import numpy as np
from mp3agent.features import vector
from mp3agent import sqlops
def build(recs,split,with_fb=True):
    X,y,g=[],[],[]
    for t,task in enumerate(recs[split]):
        row=task['row']; obs={'task':{'question':row['question']},'example':row['example'],'ordered':row['ordered']}
        pool=task['pool']; d=pool[0]
        ctx={'draft_feedback':d['feedback'],'draft_sql':d['sql'],'max_votes':max(p['votes'] for p in pool),'draft_struct':sqlops.features_of(d['sql']),'n_hints':max(1,sum(p['votes'] for p in pool))}
        for p in pool:
            X.append(vector(obs,p,p['feedback'] if with_fb else None,ctx)); y.append(int(p['correct'])); g.append(t)
    return np.array(X),np.array(y),np.array(g)
def acc(scores,y,g):
    c=0
    for t in range(g.max()+1):
        idx=np.where(g==t)[0]; c+=y[idx[np.argmax(scores[idx])]]
    return int(c)
