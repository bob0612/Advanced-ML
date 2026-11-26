"""One-off validation failure analysis (validation only, never test) of the CPU run of the augmented adapter."""
import json,sys
sys.path.insert(0,'/home/user/mp3/dev'); sys.path.insert(0,'/home/user/advanced-ml/MP3/submission')
from oracle import Split
from mp3agent.prompting import extract_sql
raw=open('/root/.claude/projects/-home-user/ccc9d568-27a5-5505-9ef2-dd156362b0e0/tool-results/mcp-runpod-exec-pod-1795627036218-ahg4ml.txt').read()
try: raw=json.loads(raw)['stdout']
except Exception: pass
raw=raw.replace('\\"','"') if raw.find('BEGIN[')<0 else raw
out=json.loads(raw[raw.find('BEGIN')+5:raw.rfind('END')])
S=Split('validation')
cats={}
for o in out:
    if o['ok']: continue
    row=S.byid[o['id']]; empty=row['example']['row_count']==0
    checked=[s[1] for s in o['trace'] if s[0]=='check' and s[1]]
    anyc=any(S.correct(row,q) for q in checked)
    llm=[extract_sql(x) for x in (o['llm'] or [])]; llm=[q for q in llm if q]
    llmc=any(S.correct(row,q) for q in llm)
    final_match=S.matches(row,o['sql'])
    key=f"empty={int(empty)} final_matches={int(final_match)} llm_called={int(bool(o['llm']))} correct_checked={int(anyc)} correct_llm={int(llmc)}"
    cats.setdefault(key,[]).append(o['id'])
for k,v in sorted(cats.items()): print(len(v),k)
print(sum(map(len,cats.values())))
