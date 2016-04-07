#!/bin/bash

# exit if not sourced
[[ "${BASH_SOURCE[0]}" != "${0}" ]] || echo 'source this script!'
[[ "${BASH_SOURCE[0]}" != "${0}" ]] || exit

# start xvfb, set display
if ! [ $(pgrep Xvfb) ] || [ ! -v DISPLAY ]; then
    killall Xvfb
    echo starting xvfb
    mkdir -p /tmp/xvfb
    Xvfb :1 -fbdir /tmp/xvfb > /tmp/xvfb_output 2>&1 &
    export DISPLAY=:1
fi
