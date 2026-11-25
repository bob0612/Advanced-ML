import json, sys
sys.path.insert(0,'/home/user/mp3/dev')
from oracle import Split, save
hints = json.load(open('/home/user/mp3/pkg/MP3_SQL_student_package/mp3_sql/data/repair_hints.json'))
for split in ['validation','train']:
    S = Split(split)
    def f(row):
        cands = [row['issue_sql']] + hints[row['id']]['suggestions']
        return [(S.matches(row,c), S.correct(row,c)) for c in cands]
    res = S.pmap(f, S.rows)
    n=len(res)
    dm=sum(r[0][0] for r in res); dc=sum(r[0][1] for r in res)
    anyc=sum(any(x[1] for x in r) for r in res)
    anym=sum(any(x[0] for x in r) for r in res)
    # policy: draft if matches else first matching hint else hint0
    def pol(r):
        for x in r:
            if x[0]: return x[1]
        return r[1][1]
    pc=sum(pol(r) for r in res)
    mc = sum(1 for r in res for x in r if x[0] and x[1]); mw=sum(1 for r in res for x in r if x[0] and not x[1])
    print(split, n, 'draft match',dm,'draft correct',dc,'any correct',anyc,'any match',anym,'policy',pc, 'match&correct',mc,'match&wrong',mw)
    save()
