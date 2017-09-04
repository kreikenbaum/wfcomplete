#! /bin/sh
### delays traffic from duckstein to bridgesrv by first parameter (or 10ms)
DELAY=$1
if [ x$DELAY = x ]; then DELAY=10; fi
sudo ~/bin/capture/delay.sh -i ens18 -d bridgesrv -m $DELAY start
ssh mkreik@bridgesrv -x "sudo ~/bin/capture/delay.sh -i ens18 -d duckstein -m $DELAY start"
