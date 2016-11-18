for i in $(cat ~/sw/top/bg.csv | shuf | head -9920 | cut -d "," -f 2); do
    . ~/bin/start_xvfb_if_necessary.sh
    echo $i
    one_site.py $i
done
