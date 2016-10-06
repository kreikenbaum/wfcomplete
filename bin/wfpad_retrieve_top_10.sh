#! /bin/bash
for i in $(head -10 ~/sw/top/top-100-modified.csv | cut -d "," -f 2); do
    . ~/bin/start_xvfb_if_necessary.sh
    echo $i
    one_site_wfpad.py $i
done
