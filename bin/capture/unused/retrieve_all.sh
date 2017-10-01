for i in $(head -10000 ~/sw/top/top-1m.csv | cut -d "," -f 2); do
    . ~/bin/start_xvfb_if_necessary.sh
    echo $i
    one_site.py $i
done
