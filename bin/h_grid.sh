#! /bin/sh
BASE=$(dirname $0)
LOG=/tmp/herr_out
# does two-step grid-search on herrmann data ($1)
# just one instance of this at the same time (overwrites /tmp/out else)
# TODO better: take all optimal values, not just the =tail= of the log
# TODO: other way to ignore the gnuplot error
env DISPLAY=:0.1 svm-grid -log2c -5,25,4 -log2g -13,1,4 -v 3 $1 | tee $LOG
C=$($BASE/log2.py $(tail -1 $LOG | cut -d ' ' -f 1) | cut -d '.' -f 1) 2>/dev/null
gamma=$($BASE/log2.py $(tail -1 $LOG | cut -d ' ' -f 2)  | cut -d '.' -f 1)
env DISPLAY=:0.1 svm-grid -log2c $(( C - 4 )),$(( C + 4 )),1 -log2g $(( gamma - 4 )),$(( gamma + 4)),1 -v 3 $1 2>/dev/null
