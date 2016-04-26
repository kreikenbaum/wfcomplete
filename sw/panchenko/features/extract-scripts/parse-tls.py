#!/usr/bin/python
# coding=utf-8
#
# Build instances from TLS data
# If specified, the records are reordered per stream, after that the records are mixed
# with respect ot start timestamp. 
#
# Additionally, one saves the number of entry nodes used and the IP of the entry node
# over which a given packet is transmitted.
#
# The ouputfile is in the form:
# [url] [start timestamp] [number of entries] [start timestamp]-[end timestamp]:[IP of entry node]:[size] [start timestamp]-[end timestamp]:[IP of entry node]:[size] ...

import sys, re, os, shutil

def exit_with_help(error=''):
	print("""\
Usage: parse-tls.py [options]

options:
	-crawlingPath { /Path/ } : Path to the Crawling Directory
	-legacy : Extracts TLS-Legacy format (default: TLS)
 """)
	print(error)
	sys.exit(1)

legacy = False
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
		elif options[i] == '-legacy':
			legacy = True
		else:
			exit_with_help('Error: Unknown Argument! ('+ options[i] + ')')
		i = i + 1


timestamppattern = re.compile('(.+) (-?\d+) (-?\d+)')

class NoTimestampLeft( Exception ):
	pass

class Packet:
	def __init__(self, flowstart=0, flowend=0, packetsize=0, srcip='', dstip=''):
		self.flowstart = int(flowstart)
		self.flowend = int(flowend)
		self.packetsize = int(packetsize)
		self.srcip = srcip
		self.dstip = dstip
		self.torip = '0.0.0.0'
    
	def __str__( self ):
		return str(self.flowstart) + '-' + str(self.flowend) + ':' + str(self.torip) + ':' + str(self.packetsize)
        
class Substream:
	def __init__(self, entry=''):
		self.entry = entry
		self.packets = []
		
class Stream:
	def __init__(self):
		pass

	def __init__(self, url='', nonce=0, timestamp=0):
		self.url = url
		self.nonce = int(nonce)
		self.timestamp = int(timestamp)
		self.entries = set()
		self.substreams = []
		self.packets = []
		self.valid = True

	def __str__( self ):
		return self.url + ' ' + str(self.timestamp/1000) + ' ' + str(len(self.entries)) + ' ' + ' '.join(['{0}'.format(el) for el in self.packets]) # timestamp / 1000 to get the precision of the filename

def mergeStreams():
	if len(currentstream.substreams) == 1:
		# we have only ane stream, we do not need to mix it
		currentstream.packets = currentstream.substreams[0].packets
	else:
		totalsumpackets = 0
		for substream in currentstream.substreams:
			totalsumpackets += len(substream.packets)
				
		# mix substreams with respect to start timestamp of the packets
		while len(currentstream.packets) < totalsumpackets:
			firstsubstreampackets = []
			for sidx, substream in enumerate(currentstream.substreams):
				if len(substream.packets) > 0:
					for idx, packet in enumerate(substream.packets):
						if idx == 0:
							firstsubstreampackets.append((sidx, packet))
							break
			(index, firstpacket) = min(firstsubstreampackets, key = lambda t: t[1].flowstart)
			currentstream.packets.append(firstpacket)
			currentstream.substreams[index].packets.pop(0)

def printCurrentStream():
	# also deletes non-printable char <feff> ([typically ?] at beginning of string) -- test it by going with arrow keys over the two '' - you will notice an additional step in the first '' which is not there in the second ''
	outputfilename = currentstream.url.replace('/', '_').replace(':', '_').replace('?', '_').replace('﻿', '')
	outputfile = open(crawlingPath + outputDir + outputfilename[:100] + '___-___' + str(currentstream.nonce), 'a')
	outputfile.write(str(currentstream) + '\n')
	outputfile.close()

# returns the next (in temporal order) time period from the timestamp file
def nexttimestamp():
	global currentstream

	while True:

		timestampline = timestampfile.readline()
		if timestampline == '':
			# EOF
			break

		match = timestamppattern.search(timestampline)

		if type(match) == type(None):
			print('Malformed line:' + timestampline)
			continue

		url = match.group(1)
		starttime = int(match.group(2)) * 1000
		nonce = int(match.group(2))
		endtime = int(match.group(3)) * 1000

		if 'check.torproject.org' in url:
			# page is uninteresting
			continue

		if (endtime < 0):
			# page wasn't loaded successfully
			continue

		if legacy and (len(currentstream.packets) > 0 and currentstream.valid):
			printCurrentStream()
		elif (len(currentstream.substreams) > 0 and currentstream.valid):
			mergeStreams()
			printCurrentStream()
		
		currentstream = Stream(url, nonce, starttime)
		return url, starttime, endtime, nonce

	raise NoTimestampLeft


if legacy:
	outputDir = 'output-tls-legacy/'
else:
	outputDir = 'output-tls/'
# Clean Output
if os.path.isdir(crawlingPath + outputDir):
	shutil.rmtree(crawlingPath + outputDir)
os.mkdir(crawlingPath + outputDir)

currentstream = Stream()
timestampfiles = os.listdir(crawlingPath + 'timestamps/')

for timestampfile in timestampfiles:

	runfile, extension = os.path.splitext(timestampfile)

	if extension != '.log':
		continue

	timestampfile = open(crawlingPath + 'timestamps/{0}.log'.format(runfile), 'r')
	dumppath = crawlingPath + 'dumps/'

	# Check if raw-to-tls has been executed before
	numberTLSfiles = sum(1 for item in os.listdir(dumppath) if os.path.isfile(os.path.join(dumppath, item)) and item.endswith('.tls'))
	if numberTLSfiles <= 0:
		print('Execute raw-to-tls.py first!')
		sys.exit(0)

	# Make sure that dump file is sorted!
	os.system('sort {0}{1}.tls -o {0}{1}.tls'.format(dumppath,runfile))

	tlsfile = open(dumppath + '{0}.tls'.format(runfile), 'r')
	ownipsfile = open(crawlingPath + 'ips/{0}.ownips'.format(runfile), 'r')
	toripsfile = open(crawlingPath + 'ips/{0}.torips'.format(runfile), 'r')

	ownips = ownipsfile.read().splitlines()
	torips = toripsfile.read().splitlines()

	try:
		url, starttime, endtime, nonce = nexttimestamp()

		for tlsline in tlsfile:

			flowstart, flowend, offset, srcip, srcport, dstip, dstport, length = tlsline.split()

			# ignore control packets (shouldn't be captured anyway)
			if srcport == '80' or dstport == '80':
				continue

			# ignore ack packets
			if length == '0':
				continue

			# negative length means error in TLS stream
			# mark current stream as invalid
			if int(length) < 0:
				currentstream.valid = False
				continue
			
			secs, usecs = flowstart.split('.')
			flowstart = int(secs) * 1000000 + int(usecs)
			sece, usece = flowend.split('.')
			flowend = int(sece) * 1000000 + int(usece)
			packet = Packet(flowstart, flowend, length, srcip, dstip)

			while (packet.flowstart > endtime):
				url, starttime, endtime, nonce = nexttimestamp()

			if (packet.flowstart < starttime):
				continue

			if not ((packet.srcip in ownips and packet.dstip in torips) or (packet.srcip in torips and packet.dstip in ownips)):
				continue

			# record the entry nodes used  
			if packet.srcip in ownips and packet.dstip in torips:
				currentstream.entries.add(packet.dstip)
				packet.torip = packet.dstip
				
			if packet.srcip in torips and packet.dstip in ownips:
				currentstream.entries.add(packet.srcip)
				packet.torip = packet.srcip

			if (packet.srcip in ownips):
				packet.packetsize = -1 * packet.packetsize
				
			if legacy:
				currentstream.packets.append(packet)
			else:
				foundsubstream = False
				if len(currentstream.substreams) > 0:
					for substream in currentstream.substreams:
						if substream.entry == packet.torip:
							foundsubstream = True
							
							# new substream order based on end of flow
							position = len(substream.packets)
							for i, stream in reversed(list(enumerate(substream.packets))):
								if packet.flowstart < stream.flowend and ((packet.packetsize<0) != (stream.packetsize<0)):
									position=i
									
								# we are too far from the actual position
								if packet.flowstart > stream.flowend:
									break
							substream.packets.insert(position, packet)
			
				if not foundsubstream:
					firstsubstream = Substream(packet.torip)
					firstsubstream.packets.append(packet)
					currentstream.substreams.append(firstsubstream)

	except NoTimestampLeft:
		pass

	# writing last stream to file
	if legacy and (len(currentstream.packets) > 0 and currentstream.valid):
		printCurrentStream()
	elif (len(currentstream.substreams) > 0 and currentstream.valid):
		mergeStreams()
		printCurrentStream()

	timestampfile.close()
	tlsfile.close()
	ownipsfile.close()
	toripsfile.close()
