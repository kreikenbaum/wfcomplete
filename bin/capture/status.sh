. config.py

echo "{
    \"local-servers\": {
        \"wtf-pad\": $(get_local_wfpad.sh),
        \"tor\": $(get_local_tor.sh),
        \"cover-traffic\": $(get_cover_server.sh)
    },
    \"bridge-servers\": {
        \"wtf-pad\": $(ssh $BRIDGE_LOGIN -x get_local_wfpad.sh),
        \"tor\": $(ssh $BRIDGE_LOGIN -x get_local_tor.sh)
    },
    \"config\": {
        \"bridge\": \"$(get_bridge.sh)\"
    },
    \"network\": {
        \"delay-to-bridgesrv\": $(get_delay.sh),
        \"delay-to-duckstein\": $(ssh $BRIDGE_LOGIN -x get_delay.sh)
    },
    \"addon\": {
        \"enabled\": $(get_addon_enabled.py),
        \"version\": $(get_version.sh),
        \"factor\":  $(get_factor.sh)
    },
    \"host\": \"$(hostname -f)\"
}"
