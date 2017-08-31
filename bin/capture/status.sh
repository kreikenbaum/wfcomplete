echo "====== BRIDGE ======="
BRIDGE=$(get_bridge.sh)
echo bridge config line:  $BRIDGE
case $BRIDGE in
    *134.169.109.51:30200*)
        echo "    direct tor at bridgesrv"
        ;;
    *127.0.0.1:30100*)
        echo "    wtf-pad client at localhost"
        ;;
    *)
        echo "    bridge unknown"
esac
echo delay: get_delay.sh
echo -e "\n====== ADDON ======="
get_addon_enabled.py
echo version: $(get_version.sh)
echo factor:  $(get_factor.sh)
