# One-off commands used on the A40 pod for the final setting (reserve 2): GPU validation, then CPU validation and test.
bash repo/MP3/experiments/eval_val.sh v3e2-r2 v3e2 '{"n":6,"temperature":1.0,"reserve":2}'
bash repo/MP3/experiments/eval_val.sh v3e2-r3 v3e2 '{"n":6,"temperature":1.0,"reserve":3}'
A=$PWD/runs/v3e2/adapter; C=$PWD/MP3_SQL_student_package/submission/config.json
bash repo/MP3/experiments/cpu_run.sh validation $C $A $PWD/runs/cpu-val-r2 0-3 &
bash repo/MP3/experiments/cpu_run.sh test $C $A $PWD/runs/cpu-test-r2 4-7 &
wait
