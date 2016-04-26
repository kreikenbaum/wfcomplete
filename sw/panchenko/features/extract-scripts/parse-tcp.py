#!/usr/bin/python
# coding=utf-8
#
# Build instances from TCP data.
# Additionally, one saves the number of entry nodes used and the IP of the entry node
# over which a given packet is transmitted.
#
# The packets over all entries are considered!
#
# The ouputfile is in the form:
# [url] [start timestamp] [number of entries] [timestamp]:[IP of entry node]:[size] [timestamp]:[IP of entry node]:[size] ...

import sys, re, os, shutil

def exit_with_help(error=''):
	print("""\
Usage: parse-tcp.py [options]

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

outputDir = 'output-tcp/'
# Clean Output
if os.path.isdir(crawlingPath + outputDir):
	shutil.rmtree(crawlingPath + outputDir)
os.mkdir(crawlingPath + outputDir)

timestamppattern = re.compile('(.+) (-?\d+) (-?\d+)')
tcpdumppattern = re.compile('(\d+)\.(\d{3})(\d+).*length (\d+)\) (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\.(\d{1,5}).* > (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})\.(\d{1,5}).* tcp (\d{1,5})')

class NoTimestampLeft( Exception ):
	pass

class Packet:
	def __init__(self, timestamp=0, packetsize=0, srcip='', dstip=''):
		self.timestamp = int(timestamp)
		self.packetsize = int(packetsize)
		self.srcip = srcip
		self.dstip = dstip
		self.torip = '0.0.0.0'
    
	def __str__( self ):
		return str(self.timestamp) + ':' + str(self.torip) + ':' + str(self.packetsize)

class Stream:
	def __init__(self):
		pass

	def __init__(self, url='', nonce=0, timestamp=0):
		self.url = url
		self.nonce = int(nonce)
		self.timestamp = int(timestamp)
		self.entries = set()
		self.packets = []

	def __str__( self ):
		return self.url + ' ' + str(self.timestamp) + ' ' + str(len(self.entries)) + ' ' + ' '.join(['{0}'.format(el) for el in self.packets])

def printCurrentStream():
	# also deletes non-printable char <feff> ([typically ?] at beginning of string) -- test it by going with arrow keys over the two '' - you will notice an additional step in the first '' which is not there in the second ''
	outputfilename = currentstream.url.replace('/', '_').replace(':', '_').replace('?', '_').replace('﻿', '')
	outputfile = open(crawlingPath + outputDir + outputfilename[:100] + '___-___' + str(currentstream.nonce), 'a')
	outputfile.write(str(currentstream) + '\n')
	outputfile.close()

# returns the next (in temporal order) time period from the timestamp file
def nexttimestamp ():
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
		starttime = int(match.group(2))
		nonce = int(match.group(2))
		endtime = int(match.group(3))
        
		if 'check.torproject.org' in url:
			# page is uninteresting
			continue
        
		if (endtime < 0):
			# page wasn't loaded successfully
			continue

		if (len(currentstream.packets) > 0):
			printCurrentStream()
		currentstream = Stream(url, nonce, starttime)

		return url, starttime, endtime, nonce
    
	raise NoTimestampLeft

currentstream = Stream()
timestampfiles = os.listdir(crawlingPath + 'timestamps/')

for timestampfile in timestampfiles:

	runfile, extension = os.path.splitext(timestampfile)

	if extension != '.log':
		continue
    
	#check if raw-to-tcp has been executed before
	if not os.path.isfile(crawlingPath + 'dumps/{0}.tcpdump'.format(runfile)):
		print('Execute raw-to-tcp.py first!')
		sys.exit(0)

	timestampfile = open(crawlingPath + 'timestamps/{0}.log'.format(runfile), 'r')
	tcpdumpfile = open(crawlingPath + 'dumps/{0}.tcpdump'.format(runfile), 'r')
	ownipsfile = open(crawlingPath + 'ips/{0}.ownips'.format(runfile), 'r')
	toripsfile = open(crawlingPath + 'ips/{0}.torips'.format(runfile), 'r')    

	ownips = ownipsfile.read().splitlines()
	torips = toripsfile.read().splitlines()

	try:
		url, starttime, endtime, nonce = nexttimestamp()

		for tcpdumpline in tcpdumpfile:

			match = tcpdumppattern.search(tcpdumpline)

			if type(match) == type(None):
				print('Malformed line:' + tcpdumpline)
				continue

			# ignore control packets (shouldn't be captured anyway)
			if match.group(6) == '80' or match.group(8) == '80':
				continue

			# ignore ack packets
			if match.group(9) == '0':
				continue

			packet = Packet(match.group(1) + match.group(2), match.group(9), match.group(5), match.group(7))

			while (packet.timestamp > endtime):
				url, starttime, endtime, nonce = nexttimestamp()

			if (packet.timestamp < starttime):
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

			currentstream.packets.append(packet)

	except NoTimestampLeft:
		pass

	# writing last stream to file
	if (len(currentstream.packets) > 0):
		printCurrentStream()

	timestampfile.close()
	tcpdumpfile.close()
	ownipsfile.close()
	toripsfile.close()
