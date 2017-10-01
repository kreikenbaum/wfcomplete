#!/bin/sh
i=10; 
while [ $i -ge 0 ]; do 
    echo $i; 
    i=$(( $i -1 )); 
    one_site test.de; 
done
