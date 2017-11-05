#! /bin/sh
pgrep -fx obfsproxy > /dev/null && echo "true" || echo "false"
