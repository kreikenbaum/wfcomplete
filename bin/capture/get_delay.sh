#! /bin/sh
### prints connection delay
DELAY=$(tc qdisc | grep ens18 | grep delay | sed 's/.*\(delay.*\)/\1/g' | cut -d ' ' -f 2)
if [ x$DELAY = x ]; then DELAY=0; fi
echo $DELAY
