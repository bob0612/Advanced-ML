# One-off driver used on the A40 pod (run from /workspace/mp3): CPU validation, then CPU test after the GPU sweep finished.
C=$PWD/MP3_SQL_student_package/submission/config.json
bash repo/MP3/experiments/cpu_run.sh validation $C $PWD/runs/v3e2/adapter $PWD/runs/cpu-val-v3e2 0-3
while ! grep -q 'SWEEP DONE' sweep.log; do sleep 20; done
bash repo/MP3/experiments/cpu_run.sh test $C $PWD/runs/v3e2/adapter $PWD/runs/cpu-test-v3e2 4-7
