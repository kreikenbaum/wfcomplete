#! /bin/sh
### checks that cover traffic server is running
if curl -s 134.169.109.25:7777?size=1; then {
    echo "true";
    exit 0
} else {
    echo "false";
    exit 1
}; fi
