import pickle, sys, numpy as np
sys.path.insert(0,'/home/user/mp3/submission'); sys.path.insert(0,'/home/user/mp3/dev')
from select_eval_lib import build, acc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
recs = pickle.load(open(sys.argv[1],'rb'))
Xt,yt,gt=build(recs,'train'); Xv,yv,gv=build(recs,'validation')
# oracle
print('oracle train',sum(yt[gt==t].max() for t in range(gt.max()+1)),'val',sum(yv[gv==t].max() for t in range(gv.max()+1)))
for C in [0.1,1]:
    m=LogisticRegression(C=C,max_iter=10000).fit(Xt,yt); print('LR',C,acc(m.decision_function(Xt),yt,gt),acc(m.decision_function(Xv),yv,gv))
for d in [2,3]:
  for lr in [0.05,0.1]:
    m=HistGradientBoostingClassifier(max_depth=d,learning_rate=lr,max_iter=200,random_state=0).fit(Xt,yt)
    print('HGB',d,lr,acc(m.predict_proba(Xt)[:,1],yt,gt),acc(m.predict_proba(Xv)[:,1],yv,gv))
m=RandomForestClassifier(500,min_samples_leaf=3,random_state=0).fit(Xt,yt); print('RF',acc(m.predict_proba(Xt)[:,1],yt,gt),acc(m.predict_proba(Xv)[:,1],yv,gv))
# breakdown by empty example for LR
m=LogisticRegression(C=1,max_iter=10000).fit(Xt,yt)
for name,X,y,g,split in [('train',Xt,yt,gt,'train'),('val',Xv,yv,gv,'validation')]:
    s=m.decision_function(X); e=[r['row']['example']['row_count']==0 for r in recs[split]]
    for flag in [True,False]:
        ts=[t for t in range(g.max()+1) if e[t]==flag]
        c=sum(y[np.where(g==t)[0][np.argmax(s[g==t])]] for t in ts); o=sum(y[g==t].max() for t in ts)
        print(name,'empty' if flag else 'nonempty',len(ts),'sel',c,'oracle',o)
