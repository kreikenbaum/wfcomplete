BRIDGE_HOST=bridgesrv
BRIDGE=mkreik@$BRIDGE_HOST
## SETUP
# tor browser
# -> tor@bridgesrv:30200
## tor on bridge
ssh $BRIDGE -x 'tor -f ~/sw/obfsproxy_wfpadtools/test/torrc.no_wfpad.server > /tmp/tor_no_wfpad.log &' &
## DO NOT delay traffic
#my_delay.sh $1
## tbb message
# better: check torrc whether activated, if not message
echo "set the line 'Bridge 134.169.109.51:30200' in the Tor Browser Bundle's settings"