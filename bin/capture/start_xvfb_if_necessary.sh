#!/bin/bash
### starts Xvfb, sets DISPLAY variable
## you can access its output as =xwud -in /tmp/xvfb_output=

# exit if not sourced
[[ "${BASH_SOURCE[0]}" != "${0}" ]] || echo 'source this script (via "source script.sh")!'
[[ "${BASH_SOURCE[0]}" != "${0}" ]] || exit 1

# check if exists
command -v Xvfb > /dev/null || echo 'Xvfb does not exist, need to install'
command -v Xvfb > /dev/null || exit 1
# start xvfb, set display
if [ ! $(pgrep Xvfb) ]; then
    echo starting xvfb
    mkdir -p /tmp/xvfb
    Xvfb :1 -fbdir /tmp/xvfb >> /tmp/xvfb_output 2>&1 &
    export DISPLAY=:1
    exit 0
fi

if [ ! -v DISPLAY ]; then
    export DISPLAY=:1
    exit 0
fi
