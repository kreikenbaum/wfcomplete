#!/usr/bin/env python

import sys
import os
# MODIFIED
import math # below import os
from datetime import datetime
# MODIFIED END
from subprocess import *

if len(sys.argv) <= 1:
	print('Usage: {0} [-log2c=begin,end,step] [-log2g=begin,end,step] [-o] [-worker=N ] [-v=N] training_file [testing_file]'.format(sys.argv[0])) ###
	###print('Usage: {0} training_file [testing_file]'.format(sys.argv[0]))
	raise SystemExit

# svm, grid, and gnuplot executable files

is_win32 = (sys.platform == 'win32')
if not is_win32:
	svmscale_exe = "../svm-scale"
	svmtrain_exe = "../svm-train"
	# MODIFIED
	svmtrain_exe_q = "../svm-train-q" # to "if not is_win32:"
	# MODIFIED END
	svmpredict_exe = "../svm-predict"
	grid_py = "./grid_patched.py" ###
	###grid_py = "./grid.py"
	gnuplot_exe = "/usr/bin/gnuplot"
else:
        # example for windows
	svmscale_exe = r"..\windows\svm-scale.exe"
	svmtrain_exe = r"..\windows\svm-train.exe"
	# MODIFIED
	svmtrain_exe_q = r"..\windows\svm-train-q.exe" # to else-case of "if not is_win32:"
	# MODIFIED END
	svmpredict_exe = r"..\windows\svm-predict.exe"
	gnuplot_exe = r"c:\tmp\gnuplot\binary\pgnuplot.exe"
	grid_py = r".\grid_patched.py" ###
	###grid_py = r".\grid.py"

assert os.path.exists(svmscale_exe),"svm-scale executable not found"
assert os.path.exists(svmtrain_exe),"svm-train executable not found"
assert os.path.exists(svmpredict_exe),"svm-predict executable not found"
###assert os.path.exists(gnuplot_exe),"gnuplot executable not found"
###assert os.path.exists(grid_py),"grid.py not found"
if not os.path.exists(gnuplot_exe): ###
	gnuplot_exe = 'null' ###
assert os.path.exists(grid_py),"grid_patched.py not found" ###
# MODIFIED
assert os.path.exists(svmtrain_exe_q),"svm-train-q executable not found" # below the block of asserts
# MODIFIED END

# MODIFIED
folds = 10 # below the block of asserts
output = False ###
# MODIFIED END
for i in range(1, len(sys.argv)): ###
	if sys.argv[i].startswith('-v='): ###
		folds = int(sys.argv[i].lstrip('-v=')) ###
	elif sys.argv[i] == '-o': ###
		output = True ###

grid_options = '' ###
if sys.argv[-2].startswith('-') or sys.argv[0] == sys.argv[-2]: ###
	testing = False ###
	train_pathname = sys.argv[-1] ###
	grid_options = ' '.join(el for el in sys.argv[1:-1]).replace('=', ' ') ###
else: ###
	testing = True ###
	train_pathname = sys.argv[-2] ###
	grid_options = ' '.join(el for el in sys.argv[1:-2]).replace('=', ' ') ###
###train_pathname = sys.argv[1]
assert os.path.exists(train_pathname),"training file not found"
file_name = os.path.split(train_pathname)[1]
scaled_file = file_name + ".scale"
model_file = file_name + ".model"
range_file = file_name + ".range"

if testing: ###
###if len(sys.argv) > 2:
	test_pathname = sys.argv[-1] ###
	###test_pathname = sys.argv[2]
	file_name = os.path.split(test_pathname)[1]
	assert os.path.exists(test_pathname),"testing file not found"
	scaled_test_file = file_name + ".scale"
	predict_test_file = file_name + ".predict"

cmd = '{0} -s "{1}" "{2}" > "{3}"'.format(svmscale_exe, range_file, train_pathname, scaled_file)
print('[########## ########] Scaling training data...')
Popen(cmd, shell = True, stdout = PIPE).communicate()

if output: ###
	cmd = 'python {0} -svmtrain "{1}" -gnuplot "{2}" {3} "{4}"'.format(grid_py, svmtrain_exe, gnuplot_exe, grid_options, scaled_file) ###
else: ###
	cmd = 'python {0} -svmtrain "{1}" -gnuplot "{2}" {3} "{4}"'.format(grid_py, svmtrain_exe_q, gnuplot_exe, grid_options, scaled_file) ###
###cmd = 'python {0} -svmtrain "{1}" -gnuplot "{2}" "{3}"'.format(grid_py, svmtrain_exe_q, gnuplot_exe, scaled_file)
# MODIFIED
time1 = datetime.now()
print('[' + str(time1).split('.')[0] + '] Cross validation...')
# MODIFIED END
f = Popen(cmd, shell = True, stdout = PIPE).stdout

line = ''
while True:
	last_line = line
	line = f.readline()
	if not line: break
c,g,rate = map(float,last_line.split())

print('[########## ########] Best c={0}, g={1} CV rate={2}'.format(c,g,rate))

# MODIFIED
time2 = datetime.now()
cExp = int(math.log(c,2))
gExp = int(math.log(g,2))
if testing: ###
###if len(sys.argv) > 2:
	cmd = '{0} -c {1} -g {2} "{3}" "{4}"'.format(svmtrain_exe, c, g, scaled_file, model_file)
	print('[' + str(time2).split('.')[0] + '] Training...')
else:
	cmd = '{0} -v {1} -c {2} -g {3} -x {4} -y {5} "{6}"'.format(svmtrain_exe, folds, c, g, cExp, gExp, scaled_file)
	print('[' + str(time2).split('.')[0] + '] Training - Cross validating...')
if not output: ###
	Popen(cmd, shell = True, stdout = PIPE).communicate()
time3 = datetime.now()
# MODIFIED END

# MODIFIED
##print('Output model: {0}'.format(model_file))
# MODIFIED END
if testing: ###
###if len(sys.argv) > 2:
	# MODIFIED
	print('[########## ########] Output model: {0}'.format(model_file)) # move behind if len(sys.argv) > 2:
	# MODIFIED END
	cmd = '{0} -r "{1}" "{2}" > "{3}"'.format(svmscale_exe, range_file, test_pathname, scaled_test_file)
	print('[########## ########] Scaling testing data...')
	Popen(cmd, shell = True, stdout = PIPE).communicate()

	cmd = '{0} "{1}" "{2}" "{3}"'.format(svmpredict_exe, scaled_test_file, model_file, predict_test_file)
	print('[' + str(datetime.now()).split('.')[0] + '] Testing...')
	Popen(cmd, shell = True).communicate()

	print('[########## ########] Output prediction: {0}'.format(predict_test_file))
# MODIFIED
else: # below the print('Output prediction: {0}'.format(predict_test_file))
	print('[########## ########] Output evaluation: {0}'.format(scaled_file + '.c_'+ str(cExp) +'_g_' + str(gExp) + '.txt'))
print('[' + str(datetime.now()).split('.')[0] + '] Time: Cross-Validation (' + str(time2-time1).split('.')[0] + '), Training (' + str(time3-time2).split('.')[0] + ')')
# MODIFIED END
