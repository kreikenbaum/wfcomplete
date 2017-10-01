#! /bin/sh
### checks that cover traffic server is running
if curl -s 134.169.109.25:7777?size=1; then {
    echo "cover traffic server active";
    exit 0
} else {
    echo "cover traffic server disabled";
    exit 1
}; fi
