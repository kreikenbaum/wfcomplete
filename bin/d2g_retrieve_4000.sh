#! /bin/bash
for i in $(cat ~/sw/4100.csv | cut -d "," -f 2); do 
    . ~/bin/start_xvfb_if_necessary.sh
    echo $i
    d2g_one_site.py $i
done
