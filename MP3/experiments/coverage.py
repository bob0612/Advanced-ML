import json, sys
sys.path.insert(0,'/home/user/mp3/dev'); sys.path.insert(0,'/home/user/mp3/submission')
from oracle import Split, save
from mp3agent.candidates import build_pool
hints = json.load(open('/home/user/mp3/pkg/MP3_SQL_student_package/mp3_sql/data/repair_hints.json'))
rw = tuple(sys.argv[1].split(',')) if len(sys.argv)>1 else ('align',)
for split in ['validation','train']:
    S=Split(split)
    def f(row):
        pool=build_pool(row['issue_sql'], hints[row['id']]['suggestions'], {'example':row['example'],'task':{'question':row['question']},'ordered':row['ordered']}, rw)
        return [(p['origin'], S.matches(row,p['sql']), S.correct(row,p['sql'])) for p in pool]
    res=S.pmap(f,S.rows)
    print(split, 'any correct', sum(any(x[2] for x in r) for r in res), 'any match&correct', sum(any(x[1] and x[2] for x in r) for r in res),
      'pool size mean', sum(len(r) for r in res)/len(res), 'max', max(len(r) for r in res))
    save()
