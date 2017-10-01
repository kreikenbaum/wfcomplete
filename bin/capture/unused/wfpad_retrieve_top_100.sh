#! /bin/bash
ssh bridgesrv 'killall tor; killall python' >> /tmp/wfpad.log 2>&1
killall python >> /tmp/wfpad.log 2>&1
~/bin/wfpad_start_servers.sh
for i in $(cat ~/sw/top/top-100-modified.csv | cut -d "," -f 2); do
    . ~/bin/start_xvfb_if_necessary.sh
    echo $i
    d2g_one_site.py $i
done
ssh bridgesrv 'killall tor; killall python' >> /tmp/wfpad.log 2>&1
killall python >> /tmp/wfpad.log 2>&1
