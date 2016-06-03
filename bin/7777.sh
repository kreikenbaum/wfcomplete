for NAME in $(ls | grep -vE '$filtered_'); do
    tshark -r $NAME -2 -R "tcp.dstport ne 7777 and tcp.srcport ne 7777" -w filtered_$NAME;
done
mkdir filtered
mv filtered_* filtered
cd filtered
for i in $(find . -name 'filtered_*'); do
    mv $i $(echo $i | sed 's/filtered_//g');
done;
xmessage "removing 7777 done at $(date)" &


