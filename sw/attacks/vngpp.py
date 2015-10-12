#Implements VNG++ classifier of Dyer et al.
#Serious details are missing from their write-up.
#1) I assume KDE is used in their Naive Bayes.
#2) I assume there are three KDEs, for time, bandwidth, burst.
#Editing out the first two gives you the base VNG. 
#Dyer et al. claims this is not as good as svm.py.

CLOSED_SITENUM = 10 #max?
CLOSED_INSTNUM = 40 #max?
OPEN_SITENUM = 40 #max?
INPUT_LOC = "data/"
OUTPUT_LOC = "vngpp.log"
CLOSED_TESTNUM = 10
OPEN_TESTNUM = 10

import subprocess
import numpy
import scipy
import scipy.stats
import math
import time

def str_to_sinste(fname):
    #given a file name fold/X-Y or fold/Z, returns (site, inst)
    #inst = -1 indicates open world
    #does not behave well with poorly formatted data

    fname = fname[fname.index("/")+1:]
    site = -1
    inst = -1
    if "-" in fname:
        fname = fname.split("-")
        site = int(fname[0])
        inst = int(fname[1])
    else:
        try:
            site = int(fname)
        except:
            site = -1
            inst = -1

    return (site, inst)

def load_one(fname):
    data = []
    try:
        f = open(fname, "r")
        lines = f.readlines()
        f.close()

        for li in lines:
            li = li.split("\t")
            t = float(li[0])
            p = int(li[1])
            data.append([t, p])
    except:
        print "Could not load", fname
    return data

def load_all():
    #writes stuff in folder into data
    #parameters are global variables

    data = [] #to return
    opendata = [] #attached to data later
    
    folder = INPUT_LOC
    cmd = "ls " + INPUT_LOC + "*"
    flist = subprocess.check_output(cmd, shell=True)
    flist = flist.split("\n")

    
    #parse flist for sorting
    fslist = []
    for f in flist:
        if "/" in f:
            (site, inst) = str_to_sinste(f)
            if (site != -1):
                fslist.append([site, inst])

    fslist = sorted(fslist, key = lambda fslist:fslist[1])
    fslist = sorted(fslist, key = lambda fslist:fslist[0])

    cursite = -1
    for f in fslist:
        (site, inst) = f
        if inst == -1:
            #open
            if site < OPEN_SITENUM:
                opendata.append(load_one(folder + str(site)))
        else:
            if inst < CLOSED_INSTNUM and site < CLOSED_SITENUM:
                if site != cursite:
                    cursite = site
                    data.append([])
                data[-1].append(load_one(folder + str(site) + "-" + str(inst)))
            #closed

    data.append([])
    for d in opendata:
        data[-1].append(d)

    return data

def get_time(sinste):
    return sinste[-1][0] - sinste[0][0]

def get_bw(sinste):
    bw = 0
    for packet in sinste:
        bw += abs(packet[1])
    return bw

def get_bursts(sinste):
    bursts = []
    totalburst = 0
    for s_i in range(0, len(sinste)):
        if s_i >= 1:
            if (sinste[s_i][1] * sinste[s_i-1][1] < 0): #change direction
                bursts.append(totalburst)
                totalburst = 0
        totalburst += abs(sinste[s_i][1])
    return bursts

def learn_kde(trainsinste):
    #INPUT: a list of sinstes from a single class
    #OUTPUT: three kdes, one for total time, one for bw, one for bursts

    times = []
    for sinste in trainsinste:
        times.append(get_time(sinste))
    time_kde = scipy.stats.gaussian_kde(scipy.array(times))

    bws = []
    for sinste in trainsinste:
        bws.append(get_bw(sinste))
    bw_kde = scipy.stats.gaussian_kde(scipy.array(bws))

    bursts = []
    for sinste in trainsinste:
        s_bursts = get_bursts(sinste)
        for b in s_bursts:
            bursts.append(b)
            
    burst_kde = scipy.stats.gaussian_kde(scipy.array(bursts))

    return [time_kde, bw_kde, burst_kde]

def logprob(size, kde):
    #kde has weird behavior with discrete things?
    #use intervals
##    prob = (kde(size-0.1) + kde(size) + kde(size+0.1)) / 3
##    prob = float(scipy.integrate.quad(kde, size-0.5, size+0.5)[0])
##    prob = max(prob, 0.00001) #don't want one packet to kill everything

    prob = max(kde(size), 0.00001)
    return math.log(prob)

data = load_all()
#data is in this format:
#each data[i] is a class
#each data[i][j] is a standard-format sequence
#standard format is: each element is a pair (time, direction)

#uses 10-fold classification

tpc = 0 #true positive counts
tnc = 0 #true negative counts
pc = 0 #positive total
nc = 0 #negative total

fout = open(OUTPUT_LOC, "w")

for fi in range(0, 10):

    #first split data into traindata and testdata.
    traindata = []
    testdata = []
    for cdata in data: #each class
        traindata.append([])
        testdata.append([])
        
        test_indices = []
        train_indices = []

        test_num_start = (len(cdata) * fi) / 10
        for ti in range(0, len(cdata)):
            if ti < test_num_start + max(len(cdata)/10, 1) and \
               ti >= test_num_start:
                test_indices.append(ti)
            else:
                train_indices.append(ti)
        
        for inst in range(0, len(cdata)):
            if inst in test_indices:
                testdata[-1].append(cdata[inst])
            else:
                traindata[-1].append(cdata[inst])

    #Training:
    #Build three KDEs for each class
    class_kdes = [] #class_kdes[i] is a list of three KDEs for class i
    for i in range(0, len(traindata)): #cycle over each class
        class_kdes.append(learn_kde(traindata[i]))

    #Testing:
    for s in range(0, len(testdata)): #cycle over each class
        for i in range(0, len(testdata[s])): #cycle over each instance
            test_sinste = testdata[s][i]

            class_probs = [] #class_probs[i] is the score of class i for this sinste
            for t in range(0, len(traindata)):
                class_probs.append(0)

            test_time = get_time(test_sinste)
            test_bw = get_bw(test_sinste)
            test_bursts = get_bursts(test_sinste)

            for sp in range(0, len(traindata)):
                class_probs[sp] += logprob(test_time, class_kdes[sp][0])
                class_probs[sp] += logprob(test_bw, class_kdes[sp][1])
                for b in test_bursts:
                    class_probs[sp] += logprob(b, class_kdes[sp][2])
                    
            gs = class_probs.index(max(class_probs)) #guessed class
            fout.write("True class: " + str(s))
            fout.write("Guessed class: " + str(gs))
            fout.write("\n")
            if s == gs:
                if s == len(testdata) - 1 : #non-monitored
                    tnc += 1
                else:
                    tpc += 1
            if s == len(testdata) - 1:
                nc += 1
            else:
                pc += 1

            print "p", tpc, "/", pc,
            print "n", tnc, "/", nc

fout.close()

print "TPR:", tpc, "/", pc
print "FPR:", tnc, "/", nc
