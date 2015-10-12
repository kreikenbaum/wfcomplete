#start with traindata, testdata

#the authors do NOT discuss what to do if several elements
#of the training set exceed the threshold

#we will therefore ignore the threshold scheme and classify to maximum score

#(something is seriously wrong with either my analysis or theirs)
CLOSED_SITENUM = 10 #max?
CLOSED_INSTNUM = 40 #max?
OPEN_SITENUM = 40 #max?
INPUT_LOC = "data/"
OUTPUT_LOC = "mnb.log"
CLOSED_TESTNUM = 10
OPEN_TESTNUM = 10

import math
import subprocess
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

def sinste_to_timing(sinste):
    #INPUT: sinste, such as data[i][j]
    #OUTPUT: timing data format, t[i] is the number of packets in the ith second

    timing = []
    totalsize = 0
    this_time = sinste[0][0]
    for i in range(0, len(sinste)):
        time = sinste[i][0]
        size = 1 #yarly. not sinste[i][1]
        
        totalsize += size
        while time - this_time > 1:
            this_time += 1
            timing.append(totalsize)
            totalsize = 0

    return timing

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

def cross_cor(timing1, timing2):
    c = 0
    length = min(len(timing1), len(timing2))
    m1 = sum(timing1)/float(length)
    m2 = sum(timing2)/float(length)
    for i in range(0, length): 
        c += (timing1[i] - m1) * (timing2[i] - m2)
    #calculate normalization
    sd1 = 0
    sd2 = 0
    for i in range(0, length):
        sd1 += (timing1[i] - m1) * (timing1[i] - m1)
        sd2 += (timing2[i] - m2) * (timing2[i] - m2)
    sd1 = math.sqrt(sd1)
    sd2 = math.sqrt(sd2)

    sd1 = max(sd1, 0.01 * m1)
    sd2 = max(sd2, 0.01 * m2)

    return c/(sd1*sd2)

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
    #We learn the timings of each sinste
    traintimings = []
    for site in traindata:
        traintimings.append([])
        for sinste in site:
            traintimings[-1].append(sinste_to_timing(sinste))

    #Testing:
    for testsite_i in range(0, len(testdata)): #for each class
        for tsinste in testdata[testsite_i]: #and each sinste
            testtiming = sinste_to_timing(tsinste)
            cross_cor_list = []
            class_list = []
            for trainsite_i in range(0, len(traintimings)):
                for traintiming in traintimings[trainsite_i]:
                    #compare timings of test and train
                    cross_cor_list.append(cross_cor(testtiming, traintiming))
                    class_list.append(trainsite_i)

            #guessed site is max of correlation
            gs = class_list[cross_cor_list.index(max(cross_cor_list))]
            #true site is this
            s = testsite_i
            
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

