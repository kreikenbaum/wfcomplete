#!/usr/bin/python
# coding=utf-8
#
# Extract TCP information from dump

import sys, os, glob

def exit_with_help(error=''):
    print("""\
Usage: raw-to-tcp.py [options]

options:
    -crawlingPath { /Path/ } : Path to the Crawling Directory
 """)
    print(error)
    sys.exit(1)

# Arguments to be read from WFP_conf
args = [ ('crawlingPath', 'dir_CRAWLING', 'crawlingPath') ]

# Checking if all variables are/will be set
for var, env, arg in args:
    if not '-'+arg in sys.argv:
        vars()[var] = os.getenv(env)
        if vars()[var] == None:
            exit_with_help('Error: Environmental Variables or Argument'+
                            ' insufficiently set! ($'+env+' / "-'+arg+'")')

# Read parameters from command line call
if len(sys.argv) != 0:
    i = 0
    options = sys.argv[1:]
    # iterate through parameter
    while i < len(options):
        if options[i] == '-crawlingPath':
            i = i + 1
            crawlingPath = options[i]
            if not crawlingPath.endswith('/'):
                crawlingPath += '/'
        else:
            exit_with_help('Error: Unknown Argument! ('+ options[i] + ')')
        i = i + 1


rawfiles = glob.glob(crawlingPath + 'dumps/*.raw')

# Process every dump
for rawfile in rawfiles:
    fullpath, extension = os.path.splitext(rawfile)

    os.system("""sudo tcpdump -r {0} -n -l -tt -q -v | sed -e 's/^[ 	]*//' | awk '/length ([0-9][0-9]*)/{{printf "%s ",$0;next}}{{print}}' > {1}""".format(rawfile, fullpath + '.tcpdump'))
