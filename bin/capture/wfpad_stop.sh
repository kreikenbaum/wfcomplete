BRIDGE_HOST=bridgesrv
BRIDGE=mkreik@$BRIDGE_HOST
kill $(ps aux | grep obfsproxy | grep -v grep | tr -s ' ' | cut -d ' ' -f 2)
ssh $BRIDGE -x 'killall tor'
ssh $BRIDGE -x 'killall python'
sudo ~/bin/capture/delay.sh -i ens18 stop
## tbb message
# better: check torrc whether activated, if message
echo "remove the line 'Bridge localhost:30100' in the Tor Browser Bundle's settings"
