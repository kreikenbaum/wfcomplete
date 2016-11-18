#!/bin/sh
# cw-param estimation
### output: 0 1 number_of_sites, training_ and testing_nstances for flearner
SITES=$(( $(ls batch | grep - | cut -d - -f 1 | sort -n | tail -1) +1 ))
INSTANCES=$(( $(ls batch | grep -v 'f$' | grep '-' | cut -d '-' -f 2 | sort  -n | uniq -c | grep "^[[:space:]]*$SITES" | tail -1 | tr -s ' ' | cut -d ' ' -f 3) +1))
TEST_I=$(( $INSTANCES /3 + 1))
TRAIN_I=$(( $INSTANCES - $TEST_I ))
echo 0 1 $SITES $TRAIN_I $TEST_I

