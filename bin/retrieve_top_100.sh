for i in $(cat ~/sw/top-100-modified.csv | cut -d "," -f 2); do 
    . ~/bin/start_xvfb_if_necessary.sh
    echo $i
    one_site.py $i
done