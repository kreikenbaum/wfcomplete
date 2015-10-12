CLOSED_SITENUM = 100 #max?
CLOSED_INSTNUM = 40 #max?
OPEN_SITENUM = 0 #max?
INPUT_LOC = "../padding/defdata2/"
OUTPUT_LOC = "svm.log"
CLOSED_TESTNUM = 10
OPEN_TESTNUM = 0

#This file simply produces the output necessary for svm_predict and svm_train
#It is essentially a feature extractor
#The only input is i from 1 to 10, which is the fold num of 10-fold classification

import sys, subprocess, math

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

##    return data

def extract(sinste):
    features = []
    
    #SIZE MARKERS
    #does not do anything for cells; see number markers
    mcount = 0 #number of markers, pad to 300 later
    sizemarker = 0 #size accumulator
    for si in range(0, len(sinste)):
        if (si > 0):
            if (sinste[si] * sinste[si-1] < 0): #direction change
                features.append(sizemarker/600)
                mcount += 1
        sizemarker += sinste[si] #can be negative
        if mcount >= 300:
            break

    for i in range(mcount, 300):
        features.append(0)

    #HTML SIZE
    #this almost certainly doesn't actually give html document size
    count_started = 0
    htmlsize = 0
    appended = 0
    for si in range(0, len(sinste)):
        if sinste[si] < 0: #incoming
            count_started = 1
            htmlsize += sinste[si]
        if sinste[si] > 0 and count_started == 1:
            features.append(htmlsize)
            appended = 1
            break
    if (appended == 0):
        features.append(0)

    #TOTAL TRANSMITTED BYTES
    totals = [0, 0]
    for si in range(0, len(sinste)):
        if (sinste[si] < 0):
            totals[0] += abs(sinste[si])
        if (sinste[si] > 0):
            totals[1] += abs(sinste[si])
    features.append(totals[0])
    features.append(totals[1])

    #NUMBER MARKERS
    mcount = 0 #also 300
    nummarker = 0
    for si in range(0, len(sinste)):
        if (si > 0):
            if (sinste[si] * sinste[si-1] < 0): #direction change
                features.append(nummarker)
                mcount += 1
        nummarker += 1
        if mcount >= 300:
            break

    for i in range(mcount, 300):
        features.append(0)

    #NUM OF UNIQUE PACKET SIZES
    uniqsizes = []
    for si in range(0, len(sinste)):
        if not(sinste[si] in uniqsizes):
            uniqsizes.append(sinste[si])
    features.append(len(uniqsizes)/2) #just 1 for cells

    #PERCENTAGE INCOMING PACKETS
    if sum(totals) != 0:
        t = totals[0]/float(sum(totals))
        t = int(t/0.05) * 0.05 #discretize by 0.05
        features.append(t)
    else:
        features.append(0)

    #NUMBER OF PACKETS
    t = totals[0] + totals[1]
    t = int(t/15) * 15 #discertize by 15
    features.append(t)

    for si in range(0, len(sinste)):
        features.append(sinste[si])

    return features

data = load_all()
#data is in this format:
#each data[i] is a class
#each data[i][j] is a standard-format sequence
#standard format is: each element is a pair (time, direction)

#uses 10-fold classification
#load foldnum from the input

try:
    fi = int(sys.argv[1])
    if fi < 1 or fi > 10:
        raise AssertionError
except:
    fi = 1
    print "python svm.py foldnum, where 1 <= foldnum <= 10"
    print "Warning: using foldnum = 1"

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

fnames = ["train", "test"]
datasets = [traindata, testdata]
for type_i in range(0, 2): #train, test
    fname = fnames[type_i]
    fout = open("svm." + fname, "w")
    for ci in range(0, len(datasets[type_i])): #class number
        for ti in range(0, len(datasets[type_i][ci])):
            features = extract(datasets[type_i][ci][ti])
            fout.write(str(ci))
            for fi in range(0, len(features)):
                fout.write(" " + str(fi+1) + ":" + str(features[fi]))
            fout.write("\n")
    fout.close()
