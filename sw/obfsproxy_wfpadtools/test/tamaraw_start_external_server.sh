#!/bin/bash

# This command starts an obfsproxy instance which is listening on
# localhost:30200 for incoming connections.  The incoming data is
# forwarded to localhost:30300 which could run a simple echo service.
# Persistent data (the server's state) is stored in
# /var/tmp/tamaraw-server.

python ../bin/obfsproxy \
	--log-min-severity=debug \
	--data-dir=/var/tmp/tamaraw-server \
	tamaraw \
	--dest 127.0.0.1:30300 \
	server 127.0.0.1:30200
