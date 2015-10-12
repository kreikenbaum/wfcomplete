#Generates 1-X from 0-X.
import math
import random

tardist = [[], []]
defpackets = []

def fsign(num):
    if num > 0:
        return 0
    else:
        return 1

def defend(list1, list2, parameter):
    datasize = 1
    mintime = 10
    
    buf = [0, 0]
    listind = 0 #marks the next packet to send
    starttime = list1[0][0]
    lastpostime = starttime
    lastnegtime = starttime
    curtime = starttime
    count = [0, 0]
    lastind = [0, 0]
    for i in range(0, len(list1)):
        if (list1[i][1] > 0):
            lastind[0] = i
        else:
            lastind[1] = i
    defintertime = [[0.02], [0.02]]
    while (listind < len(list1) or buf[0] + buf[1] > 0 or curtime < starttime + mintime):
        #print "Send packet, buffers", buf[0], buf[1], "listind", listind
        #decide which packet to send
        if (curtime >= starttime + mintime):
            for j in range(0, 2):
                if (listind > lastind[j]):
                    defintertime[j][0] = 10000
        ind = int((curtime - starttime) * 10)
        if ind >= len(defintertime[0]):
            ind = len(defintertime[0])/2
        if lastpostime + defintertime[0][ind] < lastnegtime + defintertime[1][ind]:
            cursign = 0
            curtime = lastpostime + defintertime[0][ind]
            lastpostime += defintertime[0][ind]
        else:
            cursign = 1
            curtime = lastnegtime + defintertime[1][ind]
            lastnegtime += defintertime[1][ind]
##            print "Sending packet sign", cursign, "Time", curtime, "defintertime", defintertime
##            print "Lastind", lastind
        #check if there's data remaining to be sent
        tosend = datasize
        if (buf[cursign] > 0):
            if buf[cursign] <= datasize:
                tosend -= buf[cursign]
                buf[cursign] = 0
                listind += 1
            else:
                tosend = 0
                buf[cursign] -= datasize
        if (listind < len(list1)):
            while (list1[listind][0] <= curtime and fsign(list1[listind][1]) == cursign and tosend > 0):
                if (tosend >= abs(list1[listind][1])):
                    tosend -= abs(list1[listind][1])
                    listind += 1
                else:
                    buf[cursign] = abs(list1[listind][1]) - tosend
                    tosend = 0
                if (listind >= len(list1)):
                    break
        if cursign == 0:
            list2.append([curtime, datasize])
        else:
            list2.append([curtime, -datasize])
        count[cursign] += 1
        #print count, listind
        
import sys
import os

parameters = [0, 0]
for x in sys.argv[2:]:
    parameters.append(float(x))

packets = []

parameters = [0, 1500, 0.02, 10]
    
lens = [[], []] #out, inc
deflens = [[], []]
sizes = [[], []]
defsizes = [[], []]
times = []
deftimes = []

if not os.path.exists("batchusenix-bdef"):
    os.makedirs("batchusenix-bdef")

if not os.path.exists("batch"):
    print "batch folder needs to exist"
    sys.exit(0)

for j in range(0, 100):
    print j
    for i in range(0, 90):
        packets = []
        with open("batch/" + str(j) + "-" + str(i), "r") as f:
            for x in f.readlines():
                x = x.split("\t")
                packets.append([float(x[0]), int(x[1])])
        with open("batchusenix-bdef/" + str(j) + "-" + str(i), "w") as f:
            list2 = []
            defend(packets, list2, parameters)
            list2 = sorted(list2, key = lambda list2: list2[0])
            for x in list2:
                f.write(repr(x[0]) + "\t" + repr(x[1]) + "\n")
            
