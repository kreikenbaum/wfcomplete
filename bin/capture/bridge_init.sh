. config.py
BRIDGE_LOGIN=mkreik@$BRIDGE
## SETUP
# tor browser
# -> tor@bridgesrv:30200
## tor on bridge
ssh $BRIDGE_LOGIN -x 'tor -f ~/sw/obfsproxy_wfpadtools/test/torrc.no_wfpad.server > /tmp/tor_no_wfpad.log &' &
## DO NOT delay traffic
#my_delay.sh $1
## tbb message
# better: check torrc whether activated, if not message
echo "set the line 'Bridge $BRIDGE:30200' in the Tor Browser Bundle's settings"
