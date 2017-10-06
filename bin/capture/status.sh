BRIDGE=mkreik@bridgesrv
echo "{
    \"local-servers\": {
        \"wtf-pad\": $(get_local_wfpad.sh),
        \"tor\": $(get_local_tor.sh),
        \"cover-traffic\": $(get_cover_server.sh)
    },
    \"bridge-servers\": {
        \"wtf-pad\": $(ssh $BRIDGE -x ~/bin/capture/get_local_wfpad.sh),
        \"tor\": $(ssh $BRIDGE -x ~/bin/capture/get_local_tor.sh)
    },
    \"config\": {
        \"bridge\": \"$(get_bridge.sh)\"
    },
    \"network\": {
        \"delay-to-bridgesrv\": $(get_delay.sh),
        \"delay-to-duckstein\": $(ssh $BRIDGE -x /home/mkreik/bin/capture/get_delay.sh)
    },
    \"addon\": {
        \"installed\": $(get_addon_enabled.py),
        \"version\": $(get_version.sh),
        \"factor\":  $(get_factor.sh)
    }
}"
