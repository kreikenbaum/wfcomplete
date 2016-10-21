#!/bin/sh
### output: 0 1 number_of_sites, training_ and testing_instances for flearner
SITES=$(( $(ls batch | cut -d - -f 1 | sort | tail -1) +1 ))
INSTANCES=$(( $(ls batch | grep -v 'f$' | cut -d '-' -f 2 | sort  -n | uniq -c | grep "^[[:space:]]*$SITES" | tail -1 | tr -s ' ' | cut -d ' ' -f 3) +1))
TEST_I=$(( $INSTANCES /3 + 1))
TRAIN_I=$(( $INSTANCES - $TEST_I ))
echo 0 1 $SITES $TRAIN_I $TEST_I

