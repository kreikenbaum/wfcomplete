BASE=$(dirname $0)/..
FILTER="Round"

[ "x$1" != "x" ] && cd $1

if ! [ -d batch ]; then
    $BASE/bin/wang_batch.py;
fi

if [ ! $(ls batch | sort -n | tail -20 | sort -n -k 2 -t '-' | tail -1 | grep f) ]; then
    python $BASE/sw/w/fextractor.py
fi

if [ ! -p pipe ]; then mknod pipe p; fi
(tail -f pipe | grep $FILTER > $($BASE/bin/path_filename.sh flearner.out))&
nice $BASE/sw/w/flearner $(sh $BASE/bin/flearn_params.sh) | tee pipe
killall tail
rm pipe
