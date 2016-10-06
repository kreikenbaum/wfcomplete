for i in $(tail -n +80 ~/sw/top/top-1m.csv | head -9920 | cut -d "," -f 2); do
    . ~/bin/start_xvfb_if_necessary.sh
    echo $i
    one_site.py $i
done
