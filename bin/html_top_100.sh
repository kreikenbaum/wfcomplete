# cleans cache and logs
sudo squid3 -k shutdown &&\
sudo squid3 -k kill &&\
echo '' | sudo tee /var/log/squid3/access.log &&\
sudo squid3
sleep 3
# retrieves sites
for i in $(cat ~/sw/top/top-100-modified.csv | cut -d "," -f 2); do
    mkdir $i
    cd $i
    env http_proxy='http://127.0.0.1:3128' wget $i
    cd ..
done
