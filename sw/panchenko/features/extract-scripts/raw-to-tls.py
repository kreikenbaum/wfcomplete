#!/usr/bin/python
# coding=utf-8
#
# Extract TLS information from dump

import sys, re, os, time, array, bisect, fnmatch, glob

def exit_with_help(error=''):
	print("""\
Usage: raw-to-tls.py [options]

options:
	-crawlingPath { /Path/ } : Path to the Crawling Directory
	-tcpflowPath { /Path/ } : Path to the TCPFlow Directory
 """)
	print(error)
	sys.exit(1)

# Arguments to be read from WFP_conf
args = [ ('crawlingPath', 'dir_CRAWLING', 'crawlingPath'),
         ('tcpflowPath', 'dir_BIN_TCP', 'tcpflowPath') ]

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
		elif options[i] == '-tcpflowPath':
			i = i + 1
			tcpflowPath = options[i]
			if not tcpflowPath.endswith('/'):
				tcpflowPath += '/'
		else:
			exit_with_help('Error: Unknown Argument! ('+ options[i] + ')')
		i = i + 1

namepattern = re.compile('(\d{3})\.(\d{3})\.(\d{3})\.(\d{3})\.(\d{5})-(\d{3})\.(\d{3})\.(\d{3})\.(\d{3})\.(\d{5})')

def offsetToTimestamp(offset):
    index = bisect.bisect_right(offsetlist, offset) - 1
    return mapping[offsetlist[index]]
    
def nextOffset(offset):
    index = bisect.bisect_right(offsetlist, offset)
    if index < len(offsetlist):
        return offsetlist[index]
    else:
        return None

rawfiles = glob.glob(crawlingPath + 'dumps/*.raw')

# Process every dump
for rawfile in rawfiles:
    fullpath, extension = os.path.splitext(rawfile)
    path, runfile = os.path.split(fullpath)

    if not os.path.isdir(crawlingPath + 'tmp/' + runfile):
        os.makedirs(crawlingPath + 'tmp/' + runfile)
    
    # Extract different streams
    os.system(tcpflowPath + 'src/tcpflow -r {0} -o {1}tmp/{2}'.format(rawfile, crawlingPath, runfile))

    outputfile = open(crawlingPath + 'dumps/{0}.tls'.format(runfile), 'w')

    binfiles = os.listdir(crawlingPath + 'tmp/{0}'.format(runfile))
    binfiles = fnmatch.filter(binfiles, '*[!.time]')

    for binfile in binfiles:
        match = namepattern.search(binfile)

        if type(match) == type(None):
            continue

        srcip = '%u.%u.%u.%u' % (int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4)))
        srcport = str(int(match.group(5)))
        dstip = '%u.%u.%u.%u' % (int(match.group(6)), int(match.group(7)), int(match.group(8)), int(match.group(9)))
        dstport = str(int(match.group(10)))

        timefile = open(crawlingPath + 'tmp/{0}/{1}.time'.format(runfile, binfile), 'r')
    
        mapping = {}
        offsetlist = []    

        for timeline in timefile:
            offset, sec, usec = timeline.split()
            time = '%u.%06u' % (int(sec), int(usec))
            mapping[int(offset)] = time

        offsetlist = sorted(mapping.keys())

        idlist = array.array('c')
        with open(crawlingPath + 'tmp/{0}/{1}'.format(runfile, binfile), 'rb') as f:
            while True:
                try: idlist.fromfile(f, 2000)
                except EOFError: break

        data = tuple(idlist)

        cur = 0
        prevcur = 0

        while cur is not None and (cur+4) < len(data):

            content_type = data[cur]
            version = data[cur + 1] + data[cur + 2]

            length = ord(data[cur + 3]) * 256 + ord(data[cur + 4])

            # check for valid content type
            if not content_type in ['\x14', '\x15', '\x16', '\x17']:
                
                # check for valid version
                if not version in ['\x03\x00', '\x03\x01', '\x03\x02', '\x03\x03']:
                
                    # check for valid length
                    if not length <= 16384:
                
                        # we seem to have lost one or more tcp segments
                        # mark this and the previous timestamp (i.e. tcp packet) as invalid
                        outputfile.write('%s %s %015u %s %s %s %s %u\n' % (offsetToTimestamp(prevcur), offsetToTimestamp(prevcur), prevcur, srcip, srcport, dstip, dstport, -1))
                        outputfile.write('%s %s %015u %s %s %s %s %u\n' % (offsetToTimestamp(cur), offsetToTimestamp(cur), cur, srcip, srcport, dstip, dstport, -1))
                        # jump to the next packet
                        prevcur = cur
                        cur = nextOffset(cur)
                        continue
    
            # we are only interested in application data
            if content_type == '\x17':
                # Write start and end (for reordering) of each record
                outputfile.write('%s %s %015u %s %s %s %s %u\n' % (offsetToTimestamp(cur), offsetToTimestamp(cur+4+length), cur, srcip, srcport, dstip, dstport, length))
        
            prevcur = cur
            cur = cur + 5 + length
        
    outputfile.close()
