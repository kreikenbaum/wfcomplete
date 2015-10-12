import subprocess
import math

##for c_i in range(0, 10):
##    for g_i in range(0, 10):
##        cpow = (c_i - 5) * 2
##        gpow = (g_i - 5) * 2
##        c = math.pow(10, cpow)
##        g = math.pow(10, gpow)
##        cmd = "./svm-train "
##        cmd += "-c " + str(c) + " "
##        cmd += "-g " + str(g) + " "
##        cmd += "svm.train svm.model"
##        subprocess.call(cmd, shell=True)
##
##        cmd = "./svm-predict svm.test svm.model svm.results >> temp-acc"
##        subprocess.call(cmd, shell=True)
##
##        cmd = "grep Accuracy temp-acc"
##        s = subprocess.check_output(cmd, shell=True)
##        print c_i, g_i, c, g, s
##
##        cmd = "rm svm.results"
##        cmd = "rm temp-acc"
##        subprocess.call(cmd, shell=True)

for i in range(1, 11):
    print i
    c = math.pow(10, 2)
    g = math.pow(10, -8)
    cmd = "python svm.py " + str(i)
    subprocess.call(cmd, shell=True)
    cmd = "./svm-train -c " + str(c) + " -g " + str(g) + " svm.train svm.model"
    subprocess.call(cmd, shell=True)
    
    cmd = "./svm-predict svm.test svm.model svm.results >> temp-acc"
    subprocess.call(cmd, shell=True)
    
    
