CLOSED_SITENUM = 100 #max?
CLOSED_INSTNUM = 40 #max?
OPEN_SITENUM = 0 #max?
INPUT_LOC = "../padding/defdata/"
OUTPUT_LOC = "jac.log"
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
            size = s[1]
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

def class_to_uniqcounts(datac):
    #input: all elements of a class in standard format (e.g. data[i])
    #output: dictionary, where key = packetsize, object = number of sites which had that packetsize
    count_dict = {}
    for trace in datac:
        uniqsizes = []
        for s in trace:
            size = s
            if not(size in uniqsizes):
                uniqsizes.append(size)
        for size in uniqsizes:
            if size in count_dict.keys():
                count_dict[size] += 1
            else:
                count_dict[size] = 1
                
    return count_dict

def sinste_to_uniqsizes(sinste):
    #input: one sinste (e.g. data[i][j])
    #output: list of all packet sizes that occurred in sinste
    uniqsizes = []
    for s in sinste:
        size = s
        if not(size in uniqsizes):
            uniqsizes.append(size)
    return uniqsizes

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
    #Collect the common packetsizes in each site
    class_machines = []
    #class_machines[i] is a list of packetsizes occurring in that site
    for i in range(0, len(traindata)):
        class_machines.append([])
        count_dict = class_to_uniqcounts(traindata[i])
        for k in count_dict.keys():
            if count_dict[k] > len(traindata[i])/2:
                class_machines[-1].append(k)

    #Testing:
    for s in range(0, len(testdata)): #cycle over each class
        for i in range(0, len(testdata[s])): #cycle over each instance

            class_probs = [] #class_probs[i] is the score of class i for this sinste
            for t in range(0, len(traindata)):
                class_probs.append(0)

            uniqsizes = sinste_to_uniqsizes(testdata[s][i]) #find uniqsizes
            for sp in range(0, len(traindata)): #cycle over machines
                intersec = 0 #|uniqsizes intersect class_machines[sp]|
                union = 0 #|uniqsizes union class_machines[sp]|
                for size in uniqsizes:
                    if size in class_machines[sp]:
                        intersec += 1
                    else:
                        union += 1
                union += len(class_machines[sp])
                class_probs[sp] = intersec/float(union) #jaccard's coefficient
                    
            gs = class_probs.index(max(class_probs)) #guessed site
            fout.write("True class: " + str(s))
            fout.write("Guessed class: " + str(gs))
            fout.write("\n")
            if s == gs:
                if s == len(testdata) - 1 and OPEN_TESTNUM > 0: #non-monitored
                    tnc += 1
                else:
                    tpc += 1
            if s == len(testdata) - 1 and OPEN_TESTNUM > 0:
                nc += 1
            else:
                pc += 1

fout.close()

print "TPR:", tpc, "/", pc
print "FPR:", tnc, "/", nc
