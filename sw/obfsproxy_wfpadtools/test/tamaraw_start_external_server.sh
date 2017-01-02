#!/bin/bash

# This command starts an obfsproxy instance which is listening on
# localhost:40200 for incoming connections.  The incoming data is
# forwarded to localhost:40300 which could run a simple echo service.
# Persistent data (the server's state) is stored in
# /tmp/tamaraw-server.

python ../bin/obfsproxy \
	--log-min-severity=debug \
	--data-dir=/tmp/tamaraw-server \
	tamaraw \
	--dest 127.0.0.1:40300 \
	server 127.0.0.1:40200
