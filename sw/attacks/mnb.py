CLOSED_SITENUM = 100 #max?
CLOSED_INSTNUM = 40 #max?
OPEN_SITENUM = 0 #max?
INPUT_LOC = "../padding/defdata2/"
OUTPUT_LOC = "mnb.log"
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
##    opendata = [] #attached to data later
##    
##    folder = INPUT_LOC
##    cmd = "ls " + INPUT_LOC + "*"
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
##
##    return data

def class_to_counts(datac):
    #input: all elements of a class in standard format (e.g. data[i])
    #converts standard format to counts
    #each element of counts is [packetsize, count]
    counts = []
    countsizes = []
    for trace in datac:
        for s in trace:
            size = s
            if size in countsizes:
                counts[countsizes.index(size)][1] += 1
            else:
                counts.append([size, 1])
                countsizes.append(size)
    return counts

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

def learn_kde(traincounts):
    #traincounts should come from only one class
    #returns the nb parameters for this site

    #unzip traincount
    utraincounts = []
    for c in traincounts:
        size = c[0]
        count = c[1]
        for i in range(0, count):
            utraincounts.append(size)

    utraincounts = scipy.array(utraincounts)

    #just returns gaussian

    return scipy.stats.gaussian_kde(utraincounts)

def learn_simpleprob(traincounts):
    #just converts counts to dict
    count_dict = {}
    totalfreq = 0
    for t in traincounts:
        packetsize = t[0]
        packetfreq = t[1]
        count_dict[packetsize] = packetfreq
        totalfreq += packetfreq
    for k in count_dict.keys():
        count_dict[k] /= float(totalfreq)
    return count_dict

def prob(size, machine):
    #kde has weird behavior with discrete things?
    #use intervals
##    prob = (kde(size-0.1) + kde(size) + kde(size+0.1)) / 3
    if size in machine.keys():
        prob = machine[size]
    else:
        prob = 0

    prob = max(prob, 0.00001) #pseudocount
    return prob

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
    #Rather than a KDE, each class is a simple probabilistic count
    class_machines = []
    #class_machines[i] is the machine for class i, it is a dictionary
    #class_machines[i][j] (if exists) is the probability of packet size j
    #sum of class_machines[i] should be 1
    
    for i in range(0, len(traindata)): #cycle over each class
        counts = class_to_counts(traindata[i])
        class_machines.append(learn_simpleprob(counts))

    
    #build IDF dictionary
    idf_dict = {} #also build IDF dictionary. idf_dict{packetsize} = total freq over all documents
    total_sinste = 0
    for site in traindata:
        for inst in site:
            uniqpackets = []
            for packet in inst:
                if not(packet in uniqpackets):
                    uniqpackets.append(packet)
            for packetsize in uniqpackets:
                if packetsize in idf_dict.keys():
                    idf_dict[packetsize] += 1
                else:
                    idf_dict[packetsize] = 1

            total_sinste += 1

    #Testing:
    for s in range(0, len(testdata)): #cycle over each class
        for i in range(0, len(testdata[s])): #cycle over each instance

            class_probs = [] #class_probs[i] is the score of class i for this sinste
            for t in range(0, len(traindata)):
                class_probs.append(0)

            testcounts = sinste_to_counts(testdata[s][i]) #reduce kde calls

            cosine_divider = 0
            for k in range(0, len(testcounts)):
                packetfreq = testcounts[k][1]
                cosine_divider += float(packetfreq * packetfreq)
            cosine_divier = math.sqrt(packetfreq)

            for k in range(0, len(testcounts)): #cycle over each packet count
                packetsize = testcounts[k][0]
                packetfreq = testcounts[k][1]
                packetfreq = packetfreq/cosine_divider #cosine normalization
                packetfreq = math.log(1+packetfreq) #tf
                for sp in range(0, len(traindata)):
                    p = prob(packetsize, class_machines[sp])
                    lp = math.log(p)
                    lp *= packetfreq

                    #in our case idf does nothing
                    class_probs[sp] += lp #later use min
                    
            gs = class_probs.index(max(class_probs)) #guessed site
            fout.write("True class: " + str(s))
            fout.write("Guessed class: " + str(gs))
            fout.write("\n")
            if s == gs:
                if s == len(testdata) - 1 and OPEN_SITENUM > 0 : #non-monitored
                    tnc += 1
                else:
                    tpc += 1
            if s == len(testdata) - 1 and OPEN_SITENUM > 0:
                nc += 1
            else:
                pc += 1
            print "p", tpc, "/", pc,
            print "n", tnc, "/", nc

fout.close()

print "TPR:", tpc, "/", pc
print "FPR:", tnc, "/", nc
