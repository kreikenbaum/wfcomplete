#! /bin/bash
### checks that cover traffic server is running
. config.py

if curl --connect-timeout 2 -s $MAIN:7777?size=1; then {
    echo "true";
    exit 0
} else {
    echo "false";
    exit 1
}; fi
