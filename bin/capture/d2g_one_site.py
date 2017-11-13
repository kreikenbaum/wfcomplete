#!/bin/sh

# wrapper (replaces python script)

. config.py

one_site.py $1 $BRIDGE
