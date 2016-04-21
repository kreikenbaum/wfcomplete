for NAME in $(ls | grep -vE '$filtered_'); do
    tshark -r $NAME -2 -R "tcp.dstport ne 7777 and tcp.srcport ne 7777" -w filtered_$NAME;
done
