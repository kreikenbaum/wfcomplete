CLOSED_SITENUM = 100 #max?
CLOSED_INSTNUM = 40 #max?
OPEN_SITENUM = 0 #max?
INPUT_LOC = "../padding/defdata2/"
OUTPUT_LOC = "nb.log"
CLOSED_TESTNUM = 10
OPEN_TESTNUM = 0

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
    fname = fname.split("/")[-1]
    fname = fname.split(".")[0]
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
            p = int(li)
            data.append(p)
    except:
        print "Could not load", fname
    return data

def load_all():
    #writes stuff in folder into data
    #parameters are global variables

    data = [] #to return

    #load closed world
    for i in range(0, CLOSED_SITENUM):
        data.append([])
        for j in range(0, CLOSED_INSTNUM):
            fname = str(i) + "-" + str(j) + ".size"
            data[-1].append(load_one(INPUT_LOC + fname))
    
##    folder = INPUT_LOC
##    cmd = "ls " + INPUT_LOC + "*.size"
##    flist = subprocess.check_output(cmd, shell=True)
##    flist = flist.split("\n")
##
##    
##    #parse flist for sorting
##    fslist = []
##    for f in flist:
##        if "/" in f:
##            (site, inst) = str_to_sinste(f)
##            if (site != -1):
##                fslist.append([site, inst])
##
##    fslist = sorted(fslist, key = lambda fslist:fslist[1])
##    fslist = sorted(fslist, key = lambda fslist:fslist[0])
##
##    cursite = -1
##    for f in fslist:
##        (site, inst) = f
##        if inst == -1:
##            #open
##            if site < OPEN_SITENUM:
##                opendata.append(load_one(folder + str(site)))
##        else:
##            if inst < CLOSED_INSTNUM and site < CLOSED_SITENUM:
##                if site != cursite:
##                    cursite = site
##                    data.append([])
##                data[-1].append(load_one(folder + str(site) + "-" + str(inst)))
##            #closed
##
##    data.append([])
##    for d in opendata:
##        data[-1].append(d)

    return data

def class_to_counts(datac):
    #input: all elements of a class in standard format (e.g. data[i])
    #output: dictionary, where item = packetsize, object = list of counts
    #                   for instances in that site
    count_dict = {}
    for trace in datac:
        counts = []
        countsizes = []
        for s in trace:
            size = s
            if size in countsizes:
                counts[countsizes.index(size)][1] += 1
            else:
                counts.append([size, 1])
                countsizes.append(size)
        for count in counts:
            packetsize = count[0]
            packetfreq = count[1]
            if packetsize in count_dict.keys():
                count_dict[packetsize].append(packetfreq)
            else:
                count_dict[packetsize] = [packetfreq]
                
    return count_dict

def sinste_to_counts(sinste):
    #input: one sinste (e.g. data[i][j])
    #converts standard format to counts
    #each element of counts is [packetsize, count]
    counts = []
    countsizes = []
    for s in sinste:
        size = s
        if size in countsizes:
            counts[countsizes.index(size)][1] += 1
        else:
            counts.append([size, 1])
            countsizes.append(size)
    return counts

def learn_kde(utraincounts):
    #traincounts should come from only one class
    #returns the nb parameters for this site

    #unzip traincount
##    utraincounts = []
##    for c in traincounts:
##        size = c[0]
##        count = c[1]
##        for i in range(0, count):
##            utraincounts.append(size)

    utraincounts = scipy.array(utraincounts)

    #just returns gaussian

    return scipy.stats.gaussian_kde(utraincounts)

##
##    mean = sum(utraincounts)/float(len(utraincounts))
##
##    bstd = numpy.std(utraincounts) #this is biased sample variance
##    bstd = (len(utraincounts) * bstd) /(len(utraincounts)-1)
##    return (mean, std)

def logprob(size, kde):
    #kde has weird behavior with discrete things?
    #use intervals
##    prob = (kde(size-0.1) + kde(size) + kde(size+0.1)) / 3
##    prob = float(scipy.integrate.quad(kde, size-0.5, size+0.5)[0])
##    prob = max(prob, 0.00001) #don't want one packet to kill everything

    prob = max(kde(size), 0.00001)
    return math.log(prob)

import scipy
import scipy.stats

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
    #First, populate
    #Build a KDE for each class and each packet size
    class_kdes = []
    for i in range(0, len(traindata)): #cycle over each class
        count_dict = class_to_counts(traindata[i])
        kde_dict = {}
        for packetsize in count_dict.keys():
            kde_dict[packetsize] = learn_kde(count_dict[packetsize])
        class_kdes.append(kde_dict)

    #Testing:
    for s in range(0, len(testdata)): #cycle over each class
        for i in range(0, len(testdata[s])): #cycle over each instance

            class_probs = [] #class_probs[i] is the score of class i for this sinste
            for t in range(0, len(traindata)):
                class_probs.append(0)

            testcounts = sinste_to_counts(testdata[s][i]) #reduce kde calls

            for k in range(0, len(testcounts)): #cycle over each count
                packetsize = testcounts[k][0]
                packetfreq = testcounts[k][1]
                for sp in range(0, len(traindata)):
                    if packetsize in class_kdes[sp].keys():
                        class_probs[sp] += logprob(packetfreq,
                                                   class_kdes[sp][packetsize])
                    else:
                        class_probs[sp] += math.log(0.00001)
                    
            gs = class_probs.index(max(class_probs)) #guessed site
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

fout.close()

print "TPR:", tpc, "/", pc
print "FPR:", tnc, "/", nc
