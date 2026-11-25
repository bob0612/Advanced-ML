"""Collect per-candidate records (simulating the agent's check feedback on the public DB) with labels."""
import json, sys, pickle
sys.path.insert(0,'/home/user/mp3/dev'); sys.path.insert(0,'/home/user/mp3/submission')
from oracle import Split, save
from mp3agent.candidates import build_pool
hints = json.load(open('/home/user/mp3/pkg/MP3_SQL_student_package/mp3_sql/data/repair_hints.json'))
rw = tuple(sys.argv[1].split(',')) if len(sys.argv)>1 else ('align',)
out={}
for split in ['train','validation']:
    S=Split(split)
    def f(row):
        pool=build_pool(row['issue_sql'], hints[row['id']]['suggestions'], {'example':row['example'],'task':{'question':row['question']},'ordered':row['ordered']}, rw)
        recs=[]
        for p in pool:
            r=S.public_result(row,p['sql'])
            fb={"ok":r['ok'],"columns":r['columns'],"rows":r['rows'][:8],"row_count":len(r['rows']),"truncated":len(r['rows'])>8,"error":r['error'],
                "matches_example":S.matches(row,p['sql'])}
            recs.append({**p,'origins':sorted(p['origins']),'feedback':fb,'correct':S.correct(row,p['sql'])})
        return {'row':row,'pool':recs}
    out[split]=S.pmap(f,S.rows)
    save()
pickle.dump(out,open('/home/user/mp3/dev/records_%s.pkl'%'_'.join(rw),'wb'))
print('ok')
