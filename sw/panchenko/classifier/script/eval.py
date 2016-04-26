#!/usr/bin/env python

# eval script for automation of evaluation
# supports both open and closed world evaluation
# for more information execute ./eval.py

import sys, os, math, glob, tldextract, random #, shutil
from datetime import datetime
from subprocess import *
from datetime import datetime
try:
	import tldextract
except:
	print('You need tldextract! (pip install tldextract)')
	sys.exit()

time1 = datetime.now(); time2 = time1

class EvalOption:
	def __init__(self, options):
		# eval options
		self.clean = False
		self.evaluation = True
		self.webpage = False
		self.website = False
		self.closedworld = None
		self.foreground = None
		self.foregroundList = None
		self.background = None
		self.backgroundList = None
		self.sizebg = [ 1000, 5000, 9000, 20000, 50000, 111884 ]

		self.quiet = False
		self.inputDir = os.getenv('dir_EVAL_INPUT')
		self.outputDir = os.getenv('dir_EVAL_OUTPUT')
		self.svmDir = os.getenv('dir_EVAL_LIBSVM')
		
		# grid options
		self.grid_pass_through_options = [ ] # grid
		self.gnuplot = '/usr/bin/gnuplot'

		# svm options
		self.folds = 10
		self.svm_pass_through_options = [ ] # probability, grids, weights, ...
		
		self.parse_options(options)

	def parse_options(self, options):
		if type(options) == str:
			options = options.split()
		i = 0
		
		while i < len(options):
			if options[i] == '-cleanbg':
				i = i + 1
				if not options[i] in [ '0', '1' ]:
					raise ValueError('Use 0/1 as arguments for -cleanbg.')
				else:
					if options[i] == '1':
						self.cleanbg = True
					else:
						self.cleanbg = False
			elif options[i] == '-fcw':
				i = i + 1
				self.closedworld = options[i]
			elif options[i] == '-ffg':
				i = i + 1
				self.foreground = options[i]
				self.foregroundList = 'list_' + self.foreground[0].lower() + self.foreground[1:] + '.txt'
			elif options[i] == '-fbg':
				i = i + 1
				self.background = options[i]
				self.backgroundList = 'list_' + self.background[0].lower() + self.background[1:] + '.txt'
			elif options[i] == '-fgrange':
				i = i + 1
				if ',' in options[i]:
					self.fgstart, self.fgend = list(map(int, options[i].split(',')))
					if self.fgstart > self.fgend:
						raise ValueError('Use corrent arguments for -fgrange start,end.')
				else:
					raise ValueError('Use corrent arguments for -fgrange start,end.')
			elif options[i] == '-noeval':
				self.evaluation = False
			elif options[i] == '-rmtmp':
				self.clean = True
			elif options[i] == '-bgsizes':
				i = i + 1
				if ',' in options[i]:
					self.sizebg = options[i].split(',')
					self.sizebg = list(map(int, self.sizebg))
				else:
					self.sizebg = [ int(options[i]) ]
			elif options[i] == '-webpage':
				self.webpage = True
			elif options[i] == '-website':
				self.website = True
			elif options[i] == '-dirin':
				i = i + 1
				self.inputDir = options[i]
			elif options[i] == '-dirout':
				i = i + 1
				self.outputDir = options[i]
			elif options[i] == '-q':
				self.quiet = True
			elif options[i] == '-svm':
				i = i + 1
				self.svmDir = options[i]
			elif options[i] == '-log2c':
					i = i + 1
					self.grid_pass_through_options.append('-log2c')
					self.grid_pass_through_options.append(options[i])
			elif options[i] == '-log2g':
					i = i + 1
					self.grid_pass_through_options.append('-log2g')
					self.grid_pass_through_options.append(options[i])
			elif options[i] == '-v':
				i = i + 1
				if options[i].isdigit() and int(options[i]) > 1:
					self.grid_pass_through_options.append('-v')
					self.grid_pass_through_options.append(options[i])
					self.folds = int(options[i])
				else: 
					raise ValueError('Use number n > 1 as arguments for -v.')
			elif options[i] == '-worker':
				i = i + 1
				if options[i].isdigit() and int(options[i]) > 1:
					self.grid_pass_through_options.append('-worker')
					self.grid_pass_through_options.append(options[i])
				else: 
					raise ValueError('Use number n > 1 as arguments for -worker.')
			elif options[i] == '-gnuplot':
				i = i + 1
				if options[i] == 'null':
					self.gnuplot = None
				else:
					self.gnuplot = options[i]
			else:
				self.svm_pass_through_options.append(options[i])
			i = i + 1

		#pass_through_string = ' '.join(pass_through_options)
		if not os.path.exists(self.svmDir):
			raise IOError('svm directory not found')
		else:
			# update path to binaries
			self.svm_scale = os.path.join(self.svmDir, 'svm-scale')
			self.svm_train = os.path.join(self.svmDir, 'svm-train')
			self.svm_train_q = os.path.join(self.svmDir, 'svm-train-q')
			self.grid = os.path.join(self.svmDir, 'tools/grid_patched.py')
		# Check for binaries
		if not os.path.isfile(self.svm_scale):
			raise IOError('svm-scale not found')
		if not os.path.isfile(self.svm_train):
			raise IOError('svm-train not found')
		if not os.path.isfile(self.svm_train_q):
			raise IOError('svm-train-q not found') # This way we know that svm-train is patched (!)
		if not os.path.isfile(self.grid):
			raise IOError('grid_patched.py not found') # Another indicator
		# Check gnuplot
		if self.gnuplot and not os.path.exists(self.gnuplot):
			sys.stderr.write('gnuplot executable not found\n')
			self.gnuplot = None
		# Check for input and ouput
		if not os.path.exists(self.inputDir):
			raise IOError('input directory not found')
		if not os.path.exists(self.outputDir):
			raise IOError('output directory not found')
		# Check which eval scenario 
		if self.closedworld:
			# CW Checks
			if not os.path.isfile(os.path.join(self.inputDir, self.closedworld)):
				raise IOError('input file (CW) not found')
		else: 
			# OW Checks
			if not self.foreground or not os.path.isfile(os.path.join(self.inputDir, self.foreground)):
				raise IOError('foreground file (OW) not found')
			if not self.foregroundList or not os.path.isfile(os.path.join(self.inputDir, self.foregroundList)):
				raise IOError('foreground list file (OW) not found [' + self.foregroundList + ']')
			if not self.background or not os.path.isfile(os.path.join(self.inputDir, self.background)):
				raise IOError('background file (OW) not found')
			if not self.backgroundList or not os.path.isfile(os.path.join(self.inputDir, self.backgroundList)):
				raise IOError('background list file (OW) not found [' + self.backgroundList + ']')
			if not self.webpage and not self.website:
				raise ValueError('Specify filter setting with -webpage or -website.')
			# set scenario if nothing is specified
			if not self.webpage and not self.website:
				self.website = True

if __name__ == '__main__':

	def exit_with_help():
		print("""\
Usage: eval.py [eval_options] [grid_options] [svm_options]

eval_options :
+1 -fcw name : set name that includes foreground data 
+2 -ffg name : set name that includes foreground data
              (equal number of instances per class required, no check)
        the list file includes 1 url/class per line
+2 -fbg name : set name that includes background data
        the list file includes 1 url/instance per line
*2 -fgrange start,end : set specific range for foreground evaluation
*2 -noeval : only generates evaluation data, no CV
   -rmtmp : removes all temporary files (recommended!)
         only CV and gnuplot output will be kept
*2 -bgsizes {size_1,...,size_n} : set background subsizes for evaluation
*3 -webpage : filters foreground link from background
*3 -website : filters foreground domain from background (default)
+1 sets evaluation to closed world scenario
+2 sets evaluation to open world scenario (both and list-files required)
*2 option only applicable for open world scenario
*3 option only applicable for open world scenario (one is required)

-dirin : set path that includes input files
-dirout : set path that should contain output files
-q : quiet mode (no outputs)
-svm pathname : set svm executable path and name

grid_options :
-log2c {begin,end,step | "null"} : set the range of c (default -5,15,2)
begin,end,step -- c_range = 2^{begin,...,begin+k*step,...,end}
"null"         -- do not grid with c
-log2g {begin,end,step | "null"} : set the range of g (default 3,-15,-2)
begin,end,step -- g_range = 2^{begin,...,begin+k*step,...,end}
"null"         -- do not grid with g
-v n : n-fold cross validation (default 10)
-gnuplot {pathname | "null"} :
pathname -- set gnuplot executable path and name
"null"   -- do not plot 

svm_options : additional options for svm-train
-b probability_estimates : whether to train a SVC or SVR model for 
                           probability estimates, 0 or 1 (default 0)
-wi n : set the parameter C of class i to n*C, for C-SVC (default 1)
""")
# Notes:
#  - Invalid options might be handed to libSVM and thus crash the execution.
#  - The script does not check whether the input formats in open-world evaluation are identical.
#  - Resumed grid searches biases the execution time.
#  - The open-world evaluation does not use the smaller adjusted grid as default.

#  - Gnuplot should be installed, otherwise an annoying warning will be output.
#  - The environemental variables have to be set, otherwise the script crashes.

#  - The number of foreground instances cannot be limited, in that case they should be shuffled and logged.
		sys.exit(1)

	def open_world_eval(options, filename):
		global time1; global time2
		input_file = filename
		output_file = filename
		scaled_file = output_file + '.scale'
		range_file = output_file + '.range'
		status_file = output_file + '.out'
		gnuplot_file = output_file + '.png'
		# we only print the file we are currently processing
		if not options.quiet:
			print('[' + str(datetime.now()).split('.')[0] + '] ' + os.path.relpath(input_file,options.outputDir))

		cmd = '{0} -s "{1}" "{2}" > "{3}"'.format(options.svm_scale,range_file,input_file,scaled_file)
		Popen(cmd, shell = True).communicate()

		grid_option = ' '.join(options.grid_pass_through_options)
		# check if we can resume
		if os.path.isfile(status_file):
			grid_option += ' -resume "{0}" -out "{0}" -png "{1}" '.format(status_file, gnuplot_file)
		else:
			grid_option += ' -out "{0}" -png "{1}" '.format(status_file, gnuplot_file)
		svm_option = ' '.join(options.svm_pass_through_options)
		cmd = 'python {0} -v {1} -svmtrain "{2}" -o -gnuplot "{3}" {4} {5} "{6}"'.format(options.grid,options.folds,options.svm_train,options.gnuplot,grid_option,svm_option,scaled_file)
		time1 = datetime.now()
		f = Popen(cmd, shell = True, stdout = PIPE).stdout

		line = ''
		while True:
			last_line = line
			line = f.readline()
			if not line: break
		c,g,rate = map(float,last_line.split())

		time2 = datetime.now()
		cExp = int(math.log(c,2))
		gExp = int(math.log(g,2))
		result_file = scaled_file + '.c_'+ str(cExp) +'_g_' + str(gExp) + '.txt'
		if options.clean:
			# clean output directory
			for ow_data in glob.glob(output_file+'*'):
				# do not remove result file, info file and gnuplot output
				if not ow_data in [ result_file, gnuplot_file, output_file+'-info' ]:
					os.remove(ow_data)
		return result_file
		
	def append_eval_result(options, dirpattern, current_class, info, current_result): 
		foreground = dirpattern.split('-')[1].rstrip('I')
		background = info.split('-')[0]
		tp=int(foreground)
		fn=0
		fp=0
		tn=int(background)
		for line in open(current_result, 'r'):
			if 'FP' in line:
				fp += 1
				tn -= 1
			if 'FN' in line:
				fn += 1
				tp -= 1

		result = open(os.path.join(options.outputDir, dirpattern+'-'+info), 'a')
		result.write('%04d %d %d %d %d %s\n' % (current_class, tp, fp, fn, tn, time2-time1))
		result.close()


# main code
	if len(sys.argv) < 3:
		exit_with_help()

	# parse command line options
	try:
		options = EvalOption(sys.argv[1:])
	except (IOError,ValueError) as e:
		sys.stderr.write(str(e) + '\n')
		sys.stderr.write('Try "eval.py" for more information.\n')
		sys.exit(1)

	if options.closedworld:
		# CW eval
		input_file = os.path.join(options.inputDir, options.closedworld)
		output_file = os.path.join(options.outputDir, options.closedworld)
		scaled_file = output_file + '.scale'
		range_file = output_file + '.range'
		status_file = output_file + '.out'
		gnuplot_file = output_file + '.png'

		cmd = '{0} -s "{1}" "{2}" > "{3}"'.format(options.svm_scale,range_file,input_file,scaled_file)
		if not options.quiet: 
			print('[' + str(datetime.now()).split('.')[0] + '] Scaling data...')
			Popen(cmd, shell = True, stdout = PIPE).communicate()
		else:
			Popen(cmd, shell = True).communicate()

		grid_option = ' '.join(options.grid_pass_through_options)
		# check if we can resume
		if os.path.isfile(status_file):
			grid_option += ' -resume "{0}" -out "{0}" -png "{1}" '.format(status_file, gnuplot_file)
		else:
			grid_option += ' -out "{0}" -png "{1}" '.format(status_file, gnuplot_file)
		svm_option = ' '.join(options.svm_pass_through_options)
		cmd = 'python {0} -v {1} -svmtrain "{2}" -o -gnuplot "{3}" {4} {5} "{6}"'.format(options.grid,options.folds,options.svm_train_q,options.gnuplot,grid_option,svm_option,scaled_file)
		if not options.quiet:
			print('[' + str(datetime.now()).split('.')[0] + '] Cross validation...')
		f = Popen(cmd, shell = True, stdout = PIPE).stdout

		line = ''
		while True:
			last_line = line
			line = f.readline()
			if not line: break
		c,g,rate = map(float,last_line.split())

		if not options.quiet:
			print('[########## ########] Best c={0}, g={1} CV rate={2}'.format(c,g,rate))

		cExp = int(math.log(c,2))
		gExp = int(math.log(g,2))
		result_file = scaled_file + '.c_'+ str(cExp) +'_g_' + str(gExp) + '.txt'

		if options.clean:
			# clean output directory
			for cw_data in glob.glob(output_file+'*'):
				# do not remove result file and gnuplot output
				if not cw_data in [ result_file, gnuplot_file ]:
					os.remove(ow_data)
		if not options.quiet:
			print('[########## ########] Output evaluation: {0}'.format(result_file))
	else:
		# OW eval
		
		# input files
		foreground_file = os.path.join(options.inputDir, options.foreground)
		foregroundList_file = os.path.join(options.inputDir, options.foregroundList)
		background_file = os.path.join(options.inputDir, options.background)
		backgroundList_file = os.path.join(options.inputDir, options.backgroundList)

		# database dictionaries
		linksFG={}
		namesFG={}
		domainsFG={}
		featuresFG={}
		linksBG={}
		namesBG={}
		domainsBG={}
		featuresBG={}
		
		# Process FG information
		print('[' + str(datetime.now()).split('.')[0] + '] Generating FG-Names')
		count = 1
		foreground_size = 0
		for line in open(foregroundList_file, 'r'):
			line = line.rstrip('\n')
			linksFG[count] = line
			namesFG[line] = count
			link = (line.replace('___', '://', 1)).replace('_', '/')
			domain = tldextract.extract(link)
			domainname = domain.domain
			# count different "classes" in foreground 
			foreground_size += 1
			domainsFG[count] = domainname
			count += 1
		# count instances per foreground "class"
		foreground_instances =  sum(1 for line in open(foreground_file, 'r'))/foreground_size
		
		print('[' + str(datetime.now()).split('.')[0] + '] Generating FG-Features')
		count = 1
		instance = 1
		currentClass = ''
		# Input every instance of each fg class
		for line in open(foreground_file, 'r'):
			# Every class is labeled the same as foreground (class 1)
			currentClass += '1 ' + ''.join(line.split(' ')[1:])
			if instance == foreground_instances:
				featuresFG[count] = currentClass
				instance = 1
				currentClass = ''
				count += 1
			else:
				instance += 1
		featuresFG[count] = currentClass
		
		# Process BG information
		print('[' + str(datetime.now()).split('.')[0] + '] Generating BG-Names')
		count = 1
		for line in open(backgroundList_file, 'r'):
			line = line.rstrip('\n')
			linksBG[count] = line
			namesBG[line] = count
			link = (line.replace('___', '://', 1)).replace('_', '/')
			domain = tldextract.extract(link)
			domainname = domain.domain
			domainsBG[count] = domainname
			count += 1
		
		print('[' + str(datetime.now()).split('.')[0] + '] Generating BG-Features')
		count = 1
		background_size = 0
		for line in open(background_file, 'r'):	
			# Regard all instances as background (class 0)
			featuresBG[count] = '0 ' + ''.join(line.split(' ')[1:])
			count += 1
			# count instances in background 
			background_size += 1

		dirpattern = '%04d-%03dI-%06d' % (foreground_size, foreground_instances, background_size)
		dirpath = os.path.join(options.outputDir, dirpattern)
		if not os.path.isdir(dirpath):
			os.mkdir(dirpath)
		# create evaluation files
		if options.website:
			# create Matching if necessary
			matching={}

			print('[' + str(datetime.now()).split('.')[0] + '] Generating Matching')
			for item in range(1,foreground_size+1):
				currentDomain = domainsFG[item]
				
				# List for domain already created?
				if matching.has_key(currentDomain):
					continue
				
				currentBGList = []
				for bgitem in range(1,background_size+1):
					if currentDomain == domainsBG[bgitem]:
						continue
					else:
						currentBGList.append(bgitem)
				
				matching[currentDomain] = currentBGList

			
			# Remove the complete FG domain from BG
			fgrange = range(1,foreground_size+1)
			if hasattr(options, 'fgstart') and hasattr(options,'fgend'):
				fgrange = range(options.fgstart, options.fgend+1)
			for item in fgrange:
				for size in options.sizebg:
					currentDomain = domainsFG[item]
					currentPool = matching[currentDomain]
					
					chosen = []
					if len(currentPool) < size:
						chosen = currentPool
					else: 
						chosen = random.sample(currentPool, size)
						chosen.sort()
					
					filepattern = options.foreground + '-%04d-%04d-%03dI-%06d-%06d-%06d' % (item, foreground_size, foreground_instances, len(chosen), size, background_size)
					filepath = os.path.join(dirpath, filepattern)
					
					buildFile = open(filepath + '-website', 'w')
					infoFile = open(filepath + '-website-info', 'w')
					buildFile.write(featuresFG[item])
					for instance in chosen:
						buildFile.write(featuresBG[instance])
						infoFile.write(linksBG[instance]+'\n')
					buildFile.close()
					infoFile.close()
					
					# Start LibSVM
					if options.evaluation:
						current_result = open_world_eval(options, filepath+'-website')
						append_eval_result(options, dirpattern, item, '%06d-website' % (size), current_result)
		
		if options.webpage:
			# Remove only the current page from the BG
			fgrange = range(1,foreground_size+1)
			if hasattr(options, 'fgstart') and hasattr(options,'fgend'):
				fgrange = range(options.fgstart, options.fgend+1)
			for item in fgrange:
				for size in options.sizebg:
					linkFG = linksFG[item]
					numberBG = namesBG.get(linkFG, -1)
					
					chosen = []
					currentPool = range(1,background_size+1)
					if numberBG in currentPool:
						currentPool.remove(numberBG)
					if len(currentPool) < size:
						chosen = currentPool
					else: 
						chosen = random.sample(currentPool, size)
						chosen.sort()
					
					filepattern = options.foreground + '-%04d-%04d-%03dI-%06d-%06d-%06d' % (item, foreground_size, foreground_instances, len(chosen), size, background_size)
					filepath = os.path.join(dirpath, filepattern)
					
					buildFile = open(filepath + '-webpage', 'w')
					infoFile = open(filepath + '-webpage-info', 'w')
					buildFile.write(featuresFG[item])
					for instance in chosen:
						buildFile.write(featuresBG[instance])
						infoFile.write(linksBG[instance]+'\n')
					buildFile.close()
					infoFile.close()
					
					# Start LibSVM
					if options.evaluation:
						current_result = open_world_eval(options, filepath+'-webpage')
						append_eval_result(options, dirpattern, item, '%06d-webpage' % (size), current_result)
