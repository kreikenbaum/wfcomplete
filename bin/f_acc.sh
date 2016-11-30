# as f_all.sh with class accuracy
BASE=$(dirname $0)/..
#FILTER="
FILTER=""

[ "x$1" != "x" ] && cd $1

if [ ! $(ls batch | sort -n | tail -20 | sort -n -k 2 -t '-' | tail -1 | grep f) ]; then
    python $BASE/sw/w/fextractor.py
fi

if [ ! -p pipe ]; then mknod pipe p; fi
(tail -f pipe | grep Round > $($BASE/bin/path_filename.sh flearner_acc.out))&
nice $BASE/bin/flearner_acc $(sh $BASE/bin/flearn_params.sh) | tee pipe
killall tail
rm pipex`
