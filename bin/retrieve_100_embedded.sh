mv /tmp/numEmbedded /tmp/numEmbedded.bak_$(date +%s)
for i in $(cat ~/sw/top-100-modified.csv | cut -d "," -f 2); do
    python ~/bin/htmlToNumEmbedded.py http://$i | tee -a /tmp/numEmbedded
done
