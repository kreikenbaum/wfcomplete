#! /bin/bash
### loads sites via tor browser script, capturing pcap traces
## NUM_SITES are each loaded NUM_ITERATIONS times
## if TAIL is given, use only the /last/ TAIL of these sites, else use all
if [ $# -lt 1 -o $# -gt 3 ]; then
    echo Usage: bridge_retrieve.sh NUM_SITES [NUM_ITERATIONS=50] [TAIL=NUM_SITES]
    exit 1
fi

. config.py

NUM_SITES=$1
NUM_ITERATIONS=${2:-50}
TAIL=${3:-$NUM_SITES}

for iteration in $(seq $NUM_ITERATIONS); do
    echo -e "======= ITERATION: $iteration =============\n"
    for site in $(head -$NUM_SITES $SITES | tail -$TAIL | cut -d "," -f 2); do
        . start_xvfb_if_necessary.sh
        echo $site
        d2g_one_site.py $site
    done
done
