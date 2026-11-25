import pickle, sys, numpy as np
sys.path.insert(0,'/home/user/mp3/submission')
from mp3agent.features import vector, NAMES
from mp3agent import sqlops
from sklearn.linear_model import LogisticRegression
recs = pickle.load(open(sys.argv[1],'rb'))
def build(split):
    X,y,g=[],[],[]
    for t,task in enumerate(recs[split]):
        row=task['row']; obs={'task':{'question':row['question']},'example':row['example'],'ordered':row['ordered']}
        pool=task['pool']; d=pool[0]
        ctx={'draft_feedback':d['feedback'],'draft_sql':d['sql'],'max_votes':max(p['votes'] for p in pool),'draft_struct':sqlops.features_of(d['sql']),
             'n_hints':max(1,sum(p['votes'] for p in pool))}
        for p in pool:
            X.append(vector(obs,p,p['feedback'],ctx)); y.append(int(p['correct'])); g.append(t)
    return np.array(X),np.array(y),np.array(g)
def acc(scores,y,g):
    n=g.max()+1; c=0
    for t in range(n):
        idx=np.where(g==t)[0]; c+=y[idx[np.argmax(scores[idx])]]
    return c,n
Xt,yt,gt=build('train'); Xv,yv,gv=build('validation')
# heuristic: match first, then hint votes, then pool order
def heur(X):
    i={n:k for k,n in enumerate(NAMES)}
    return X[:,i['match']]*100 + X[:,i['col_exact']]*10 + X[:,i['votes']] - X[:,i['o_draft']]*0.5
print('heur train',acc(heur(Xt),yt,gt),'val',acc(heur(Xv),yv,gv))
for C in [0.03,0.1,0.3,1,3]:
    m=LogisticRegression(C=C,max_iter=5000).fit(Xt,yt)
    print('LR C',C,'train',acc(m.decision_function(Xt),yt,gt),'val',acc(m.decision_function(Xv),yv,gv))
m=LogisticRegression(C=1,max_iter=5000).fit(Xt,yt)
for n,w in sorted(zip(NAMES,m.coef_[0]),key=lambda x:-abs(x[1])): print(f'{n:22s}{w:+.2f}')
