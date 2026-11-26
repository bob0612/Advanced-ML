#!/bin/bash
# usage: ablate.sh name split [train-args] -- [jq-like python edits]
cd /home/user/mp3/pkg/MP3_SQL_student_package && . .venv/bin/activate
NAME=$1; SPLIT=$2; TRAIN_ARGS=$3; EDIT=$4
mkdir -p /home/user/mp3/ablations
CFG=/home/user/mp3/ablations/$NAME.json
if [ -n "$TRAIN_ARGS" ]; then PYTHONPATH="$PWD:$PWD/submission" python submission/tools/train_ranker.py --output $CFG $TRAIN_ARGS >/dev/null; else cp submission/config.json $CFG; fi
[ -n "$EDIT" ] && python -c "import json;c=json.load(open('$CFG'));$EDIT;json.dump(c,open('$CFG','w'),indent=1)"
PYTHONPATH="$PWD/submission" python -m mp3_sql.predict --split $SPLIT --method submission/submission.py --checkpoint $CFG --output /home/user/mp3/ablations/$NAME-$SPLIT.json >/dev/null
python -m mp3_sql.score --split $SPLIT --predictions /home/user/mp3/ablations/$NAME-$SPLIT.json --output /home/user/mp3/ablations/$NAME-$SPLIT-score.json | python -c "import json,sys;d=json.load(sys.stdin);print('$NAME',d['correct'],'/',d['n'],round(d['accuracy'],2),'calls',round(d['mean_tool_calls'],2))"
