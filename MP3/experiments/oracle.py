"""Dev-only: evaluate SQL on public example and on all 3 grading instances (train/validation only)."""
import json, sys, hashlib, pickle, os, threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
PKG = Path('/home/user/mp3/pkg/MP3_SQL_student_package')
sys.path.insert(0, str(PKG))
from mp3_sql.common import load_bank, result_fingerprint
from mp3_sql.sqlite_tool import execute_sql

CACHE_F = Path('/home/user/mp3/dev/cache.pkl')
_lock = threading.Lock()
try: CACHE = pickle.loads(CACHE_F.read_bytes())
except Exception: CACHE = {}
def save():
    with _lock: CACHE_F.write_bytes(pickle.dumps(CACHE))

class Split:
    def __init__(self, split):
        assert split in ('train', 'validation')
        self.rows, self.manifest, self.folder = load_bank(split)
        self.byid = {r['id']: r for r in self.rows}
        self.refs = json.loads((self.folder / f'{split}_references.json').read_text())['tasks']
        self.public = json.loads((self.folder / 'public_results.json').read_text())
    def run(self, path, row, sql):
        key = (str(path), sql)
        if key in CACHE: return CACHE[key]
        tables = [t['name'] for t in self.manifest['databases'][row['db_id']]['schema']]
        r = execute_sql(self.folder / path, sql, tables)
        with _lock: CACHE[key] = r
        return r
    def public_result(self, row, sql):
        return self.run(self.manifest['databases'][row['db_id']]['public_path'], row, sql)
    def matches(self, row, sql):
        r = self.public_result(row, sql)
        return bool(r['ok'] and result_fingerprint(r, row['ordered']) == self.public[row['id']])
    def correct(self, row, sql):
        for case in self.refs[row['id']]:
            r = self.run(case['path'], row, sql)
            if not (r['ok'] and result_fingerprint(r, row['ordered']) == case['expected']): return False
        return True
    def pmap(self, fn, items, workers=8):
        with ThreadPoolExecutor(workers) as p: return list(p.map(fn, items))
