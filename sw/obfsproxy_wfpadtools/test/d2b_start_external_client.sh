#!/bin/bash

# This command starts an obfsproxy instance which listens for SOCKS connections
# on localhost:30100.  Incoming SOCKS data is then forwarded to the
# ScrambleSuit server running on localhost:30200.  Persistent data (the
# client's session ticket) is stored in /var/tmp/wfpad-client.

#python /usr/local/bin/obfsproxy \
python  ../bin/obfsproxy \
	--log-min-severity=debug \
	--data-dir=/var/tmp/wfpad-client \
	wfpad \
	--dest 134.169.109.51:30200 \
	client 127.0.0.1:30100
