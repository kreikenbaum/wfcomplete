#!/bin/bash

# This command starts an obfsproxy instance which listens for SOCKS connections
# on localhost:40100.  Incoming SOCKS data is then forwarded to the
# ScrambleSuit server running on localhost:40200.  Persistent data (the
# client's session ticket) is stored in /tmp/wfpad-client.

#python /usr/local/bin/obfsproxy \
python  ../bin/obfsproxy \
	--log-min-severity=debug \
	--data-dir=/tmp/wfpad-client \
	tamaraw \
	--dest 134.169.109.51:40200 \
	client 127.0.0.1:40100
