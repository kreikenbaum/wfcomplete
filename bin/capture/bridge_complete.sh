#! /bin/bash
### captures traces using bridge_retrieve.sh, afterwards creates dir,
### converts them, etc
## NUM_SITES are each loaded NUM_ITERATIONS times
## if TAIL is given, use only the /last/ TAIL of these sites, else use all
if [ $# -lt 1 -o $# -gt 3 ]; then
    echo Usage: bridge_complete.sh NUM_SITES [NUM_ITERATIONS=50] [TAIL=NUM_SITES]
    exit 1
fi

NUM_ITERATIONS=${2:-50}
TAIL=${3:-$NUM_SITES}

. config.py

status.sh > $SAVETO/status
NAME=$(check_name.py $TAIL $NUM_ITERATIONS) || (echo "status invalid"; exit 1)

bridge_retrieve.sh $1 $2 $3

cd $SAVETO && mv -i status *@* $NAME && echo traces are at $NAME
(cd $NAME && \
     counter.py > /tmp/counter_out_$(date +%F:%T) 2>&1 && \
     zip pagetext.zip $(ls | grep -E '\.text') && \
     mkdir -p $SAVETO/skip/last_traces && \
     mv -b *@* $SAVETO/skip/last_traces
) &
