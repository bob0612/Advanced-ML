# One-off driver used on the A40 pod: cross-fit reserve sweep, one setting at a time (parallel runs ran out of GPU memory).
for r in 2 3 4 5; do bash repo/MP3/experiments/crossfit_eval.sh r$r "{\"reserve\": $r}"; done; echo SWEEP2 DONE
