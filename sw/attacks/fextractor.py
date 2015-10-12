def extract(times, sizes, features):

    #Transmission size features
    features.append(len(sizes))

    count = 0
    for x in sizes:
        if x > 0:
            count += 1
    features.append(count)
    features.append(len(sizes)-count)

    features.append(sizes[-1] - sizes[0])

    #Unique packet lengths
    for i in range(-1500, 1501):
        if i in sizes:
            features.append(1)
        else:
            features.append(0)

    #Transpositions (similar to good distance scheme)
    count = 0
    for i in range(0, len(sizes)):
        if sizes[i] > 0:
            count += 1
            features.append(i)
        if count == 300:
            break
    for i in range(count, 300):
        features.append("X")
        
    count = 0
    prevloc = 0
    for i in range(0, len(sizes)):
        if sizes[i] > 0:
            count += 1
            features.append(i - prevloc)
            prevloc = i
        if count == 300:
            break
    for i in range(count, 300):
        features.append("X")


    #Packet distributions (where are the outgoing packets concentrated)
    for i in range(0, min(len(sizes), 3000)):
        if i % 30 != 29:
            if sizes[i] > 0:
                count += 1
        else:
            features.append(count)
            count = 0
    for i in range(len(sizes)/30, 100):
        features.append(0)

    #Bursts
    bursts = []
    curburst = 0
    stopped = 0
    for x in sizes:
        if x < 0:
            stopped = 0
            curburst -= x
        if x > 0 and stopped == 0:
            stopped = 1
        if x > 0 and stopped == 1:
            stopped = 0
            bursts.append(curburst)
    if (len(bursts) > 0):
        features.append(max(bursts))
        features.append(sum(bursts)/len(bursts))
        features.append(len(bursts))
    else:
        features.append("X")
        features.append("X")
        features.append("X")
    counts = [0, 0, 0]
    for x in bursts:
        if x > 5:
            counts[0] += 1
        if x > 10:
            counts[1] += 1
        if x > 15:
            counts[2] += 1
    features.append(counts[0])
    features.append(counts[1])
    features.append(counts[2])
    for i in range(0, 5):
        try:
            features.append(bursts[i])
        except:
            features.append("X")

    for i in range(0, 20):
        try:
            features.append(sizes[i] + 1500)
        except:
            features.append("X")

#fold = "batch/"

import sys
import subprocess

if len(sys.argv) < 2:
    print "python fextractor.py <folder>"
    sys.exit(0)
else:
    fold = sys.argv[1]
    
a = subprocess.check_output("ls " + fold + "*.size", shell=True)
flist = a.split("\n")

count = 0
for fname in flist:
    count += 1
    if (count % 100 == 0):
        print count
    try: 
        f = open(fname, "r")
    ##        f = open(fold + fname + ".cell", "r")
        
        #Set up times, sizes
        times = []
        sizes = []
        for x in f:
            sizes.append(int(x))
##            x = x.split("\t")
##            times.append(float(x[0]))
##            sizes.append(int(x[1]))
        f.close()

        #Extract features. All features are non-negative numbers or X. 
        features = []
        extract(times, sizes, features)

        fout = open(fname + "f", "w")
        for x in features:
            fout.write(repr(x) + " ")
        fout.close()
    except:
        pass


f = open("fdetails", "w")
f.write(str(len(features)))
print len(features)
f.close()
