#! /bin/bash
pgrep -x tor > /dev/null && echo "true" || echo "false"
