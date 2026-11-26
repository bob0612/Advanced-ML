import resource, subprocess, time, sys, os
t=time.perf_counter()
env=dict(os.environ, PYTHONPATH=os.getcwd()+"/submission")
pre=resource.getrusage(resource.RUSAGE_CHILDREN)
for cmd in [["python","-m","mp3_sql.predict","--split","test","--method","submission/submission.py","--checkpoint","submission/config.json","--output","outputs/test.json"],
            ["python","-m","mp3_sql.score","--split","test","--predictions","outputs/test.json","--output","outputs/test-score.json"]]:
    subprocess.run(["taskset","-c","0-3","bash","-c","ulimit -v 2097152; exec "+" ".join(cmd)],check=True,env=env,stdout=subprocess.DEVNULL)
r=resource.getrusage(resource.RUSAGE_CHILDREN)
print({"wall_s":round(time.perf_counter()-t,1),"user_cpu_s":round(r.ru_utime-pre.ru_utime,1),"sys_cpu_s":round(r.ru_stime-pre.ru_stime,1),"max_child_rss_MiB":round(r.ru_maxrss/1024,1)})
