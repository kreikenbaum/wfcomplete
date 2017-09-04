BRIDGE=mkreik@bridgesrv
echo ====== LOCAL ============
printf "wtf-pad: "
get_local_wfpad.sh
printf "tor: "
get_local_tor.sh
echo ====== BRIDGESRV ========
printf "wtf-pad: "
ssh $BRIDGE -x ~/bin/capture/get_local_wfpad.sh
printf "tor: "
ssh $BRIDGE -x ~/bin/capture/get_local_tor.sh
echo ====== BRIDGE ===========
BS=$(get_bridge.sh)
echo bridge config line:  $BS
case $BS in
    *134.169.109.51:30200*)
        echo "    direct tor at bridgesrv"
        ;;
    *127.0.0.1:30100*)
        echo "    wtf-pad client at localhost"
        ;;
    *)
        echo "    bridge unknown"
esac
echo delay to bridgesrv: $(get_delay.sh)
echo delay to duckstein: $(ssh $BRIDGE -x /home/mkreik/bin/capture/get_delay.sh)
echo ====== ADDON ============
get_addon_enabled.py
echo version: $(get_version.sh)
echo factor:  $(get_factor.sh)
