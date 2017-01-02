#!/bin/bash

# This command starts an obfsproxy instance which is listening on
# localhost:40200 for incoming ScrambleSuit connections.  The incoming data is
# then deobfuscated and forwarded to localhost:40300 which could run a simple
# echo service.  Persistent data (the server's state) is stored in
# /tmp/scramblesuit-server.

python ../bin/obfsproxy \
	--log-min-severity=debug \
	--data-dir=/tmp/wfpad-server \
	wfpad \
	--dest 127.0.0.1:40300 \
	server 0.0.0.0:40200
