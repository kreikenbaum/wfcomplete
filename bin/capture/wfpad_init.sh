BRIDGE_HOST=bridgesrv
BRIDGE=mkreik@$BRIDGE_HOST
OBFSPROXY=~/sw/obfsproxy_wfpadtools/bin/obfsproxy
## SETUP
# tor browser
# -> wfpad-client@duckstein:30100
# -> wfpad-server@bridgesrv:30200
# -> tor@bridgesrv:30300
# tc delays all traffic to bridgesrv
## wfpad-client
python  $OBFSPROXY \
        --log-min-severity=debug \
        --data-dir=/tmp/wfpad-client \
        wfpad \
        --dest 134.169.109.51:30200 \
        client 127.0.0.1:30100 \
        > /tmp/wfpad-client.log &
## wfpad-server on bridge
ssh $BRIDGE -x "python ~/sw/obfsproxy_wfpadtools/bin/obfsproxy --log-min-severity=debug --data-dir=/tmp/wfpad-server wfpad --dest 127.0.0.1:30300 server 0.0.0.0:30200  >/tmp/wfpad-server.log &" &
## tor on bridge
ssh $BRIDGE -x 'tor -f ~/sw/obfsproxy_wfpadtools/test/torrc.server > /tmp/tor.log &' &
## delay traffic
my_delay.sh $1
## tbb message
# better: check torrc whether activated, if not message
echo "set the line 'Bridge 127.0.0.1:30100' in the Tor Browser Bundle's settings"







