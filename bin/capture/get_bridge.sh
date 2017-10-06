B=$(grep 'Bridge ' ~/bin/tor-browser_en-US/Browser/TorBrowser/Data/Tor/torrc)
case $B in
    *134.169.109.51:30200*)
        echo "$B (direct tor at bridgesrv)"
        ;;
    *127.0.0.1:30100*)
        echo "$B (wtf-pad client at localhost)"
        ;;
    *)
        echo "$B (bridge unknown)"
esac

