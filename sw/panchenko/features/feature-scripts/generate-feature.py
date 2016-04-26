#!/usr/bin/python

# Script to generate Features from previously prepared data
# Outlier Removal should be done beforehand (!)
#
# python generate-feature.py [options]

import sys, os, glob, itertools, numpy, random, shutil, errno
try:
    from natsort import natsorted
except:
    print('You need natsort! (pip install natsort)')
    sys.exit()

def exit_with_help(error=''):
    print("""\
Usage: generate-feature.py [options]

options:
   -in { /Path/ } : Path to Outlierfree Inst. (with Formats as Sub-Dirs)
   -out { /Path/ } : Path to Generated Features (prob. creates Sub-Dirs)
   -force { YES | NO } : Uses {format} instead of {format}-outlierfree
                                                    as Input Directories

   -dataSet { _ | \Name\ } : If \Name\ is given, Features are Outputted
                               in a Single File otherwise in Directories
   -instances { #Number } : Number (!) of Instances per Webpage
   -randomInstances { YES | NO } : Take Random Instances if Possible
   -setting { CW | OW_BG | OW_FG } : Evaluated Scenario 
                                     (Class Labeling differs)
   -classifier { CUMULATIVE | SEPARATE } : Used Classifier Version to 
                                                     Generate Features
   -features { #Number } : Number of Interpolated Features

Default Configuration is extracted from Environmental Variables
        Check, Adjust & Reload WFP_config if necessary

Notes:
 - Execute outlier-removal.py before calculating Features! (!!!)
 - Instances that break the Classifier are ALWAYS removed!
 - Script always processes ALL available formats!
 - Do NOT have more instances as input than specified by numOfInstances...
 - Background-Setting assumes a single instance per webpage 
   if randomInstances is set. This WILL BREAK synchronizedness
 """)
    print(error)
    sys.exit(1)

# Define formats
# Define formats (input, output, abbreviation)
formats = [ ('tcp', 'TCP'), ('tls', 'TLS'), ('tls-legacy', 'TLSLegacy'),
            ('tls-nosendme', 'TLSNoSendMe'), ('tls-nosendme-legacy', 'TLSNoSendMeLegacy'),
            ('cell', 'Cell'), ('cell-legacy', 'CellLegacy'),
            ('cell-nosendme', 'CellNoSendMe'), ('cell-nosendme-legacy', 'CellNoSendMeLegacy') ]

tmp4 = 'NO'
# Arguments to be read from WFP_conf
args = [ ('outlierfreePath', 'dir_TEMP_OUTLIERFREE', 'in'),
         ('featurePath', 'dir_FETCH_FEATURES', 'out'),
         ('setting', 'conf_SETTING', 'setting'),
         ('dataSet', 'conf_DATASET', 'dataSet'),
         ('tmp1', 'conf_CLASSIFIER', 'classifier'),
         ('tmp2', 'conf_RANDOM_INSTANCES', 'randomInstances'),
         ('tmp3', 'conf_INSTANCES', 'instances'),
         #('tmp4', 'conf_FORCE', 'force'),
         ('tmp5', 'conf_FEATURES', 'features') ]

# Checking if all variables are/will be set
for var, env, arg in args:
    if not '-'+arg in sys.argv:
        vars()[var] = os.getenv(env)
        if vars()[var] == None:
            exit_with_help('Error: Environmental Variables or Argument'+
                        ' insufficiently set! ($'+env+' / "-'+arg+'")')

# Read parameters from command line call
if len(sys.argv) != 0:
    i = 0
    options = sys.argv[1:]
    # iterate through parameter
    while i < len(options):
        if options[i] == '-in':
                i = i + 1
                outlierfreePath = options[i]
        elif options[i] == '-out':
                i = i + 1
                featurePath = options[i]
        elif options[i] == '-setting':
                i = i + 1
                setting = options[i]
        elif options[i] == '-dataSet':
                i = i + 1
                dataSet = options[i]
        elif options[i] == '-classifier':
                i = i + 1
                tmp1 = options[i]
        elif options[i] == '-randomInstances':
                i = i + 1
                tmp2 = options[i]
        elif options[i] == '-instances':
                i = i + 1
                tmp3 = options[i]
        elif options[i] == '-force':
                i = i + 1
                tmp4 = options[i]
        elif options[i] == '-features':
                i = i + 1
                tmp5 = options[i]
        else:
            exit_with_help('Error: Unknown Argument! ('+ options[i] + ')')
        i = i + 1

# Check set variables
if not os.path.isdir(outlierfreePath):
    exit_with_help('Error: Invalid Input Path!')
if setting not in [ 'CW', 'OW_BG', 'OW_FG' ]:
    exit_with_help('Error: Unknown Setting!')
if tmp1 in [ 'CUMULATIVE', 'SEPARATE' ]:
    if tmp1 == 'SEPARATE':
        separateClassifier = True
    else:
        separateClassifier = False
else:
    exit_with_help('Error: Unknown Classifier Option!')
if tmp2 in [ 'YES', 'NO' ]:
    if tmp2 == 'YES':
        randomInstances = True
    else:
        randomInstances = False
else:
    exit_with_help('Error: Unknown Random Instances Option!')
if tmp3.isdigit():
    if int(tmp3) > 0:
        numOfInstances = int(tmp3)
    else:
        exit_with_help('Error: Number of Instances is not a Number!')
else:
    exit_with_help('Error: Number of Instances is not a Number!')
if tmp4 in [ 'YES', 'NO' ]:
    if tmp4 == 'YES':
        force = True
    else:
        force = False
else:
    exit_with_help('Error: Unknown Force Option!')
if tmp5.isdigit():
    if int(tmp5) % 2 == 0:
        featureCount = int(tmp5)
    else:
        exit_with_help('Error: Number of Instances is not even!')
else:
    exit_with_help('Error: Number of Features is not a Number!')

# Additional checks
if setting == 'OW_BG' and '-force' not in sys.argv:
    outlierRemoval = 'None'
if setting == 'OW_BG':
    numOfInstances = 1

# For reading instances
class Instance:
    def __init__(self):
        pass
    
    def __init__(self, url='', timestamp=0, entries=0, packets=[]):
        self.url = url
        self.timestamp = int(timestamp)
        self.entries = int(entries)
        self.packets = packets
        
# For assigning packets transmitted over a certain entry node
class EntryNodePacket:
    def __init__(self):
        pass
        
    def __init__(self, entry=''):
        self.entry = str(entry)
        self.packets = []
        
# For reading packets from instances
class Packet:
    def __init__(self):
        pass
        
    def __init__(self, torip='', packetsize=''):
        self.torip = str(torip)
        self.packetsize = int(packetsize)

if dataSet == '_':
    # Clean up output folder
    if os.path.isdir(featurePath):
        shutil.rmtree(featurePath)

try:
    os.mkdir(featurePath)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

if dataSet != '_':
    # Remove output files
    for _, form in formats:
        try:
            os.remove(featurePath+dataSet+'_'+form)
            os.remove(featurePath+'list_'+dataSet[0].lower()+dataSet[1:]+'_'+form+'.txt')
        except OSError as e:
            if e.errno != errno.ENOENT:
                raise

current=0
countList = [0] * (len(formats))
maxList = [0] * (len(formats))
for directory, nameDir in formats:
    
    inDir = 'output-'+directory
    outDir = 'feature-'+directory
    
    currentpath = outlierfreePath+inDir+'-outlierfree/'
    
    # skip non existing
    if not os.path.isdir(currentpath):
        current += 1
        currentpath = outlierfreePath+inDir+'/'
        if os.path.isdir(currentpath):
            if not force:
                print('WARN: Only outlier-free formats will be used! ('+nameDir+')')
                print('      Look into force option for different behavior.')
                continue
            else:
                print('INFO: Directory without outlier-removal will be used! ('+nameDir+')')
                print('      Look into force option for different behavior.')
        else:
            continue
    
    # determine the class labeling
    if setting in [ 'CW', 'OW_FG' ]:
        classLabel = 1
    else:
        classLabel = 0
    
    inFiles = natsorted(glob.glob(currentpath+'*'))
    maxList[current] = len(inFiles)
    
    count = 1
    
    # Create Single Output files if wished with info file
    if dataSet != '_':
        fdout = open(featurePath + dataSet + '_' + nameDir, 'w')
        fdoutInfo = open(featurePath + 'list_' + dataSet[0].lower() + dataSet[1:] + '_' + nameDir + '.txt', 'w')
    else:
        os.mkdir(featurePath + outDir + '/')
    
    for inFile in inFiles:
        filename = os.path.split(inFile)[1]
        
        #print 'INFO: (' + str(count) + '/' + str(len(inFiles)) + ') ' +  filename
        
        fdin = open(inFile, 'r')
        # Create Output files for each page
        if dataSet == '_':
            fdout = open(featurePath + outDir + '/' + filename, 'w')
        
        instances = []
        
        # Read every available instance
        for instanceline in fdin:
            entries = []
            packets = []

            entries.extend(instanceline.rstrip().split(' '))
            incomingsize = 0
            
            url = entries[0]
            timestamp = entries[1]
            # for compatibility 
            if len(timestamp) > 13:
                border = len(timestamp) - 13
                timestamp = timestamp[:-border]
            if ':' in entries[2]:
                entrynodes = '0'
                data = entries[2:]
            else:
                entrynodes = entries[2]
                data = entries[3:]
            
            # Split time and data, add outlier criterium (not implemented!)
            for entry in data:
                # for compatibility
                if len(entry.split(':')) == 3:
                    packet = Packet(entry.split(':')[1], entry.split(':')[2])
                elif len(entry.split(':')) == 2:
                    packet = Packet('', entry.split(':')[1])
                else:
                    print "ERROR: Unkown instance format!"
                    packet = Packet('', '')
                packets.append(packet)
                size = packet.packetsize
                if size > 0:
                    incomingsize += size
            
            instance = Instance(url, timestamp, entrynodes, packets)
            instances.append((incomingsize, instance))
            
            # Only use a single instance for background
            if setting == 'OW_BG':
                break
        
        fdin.close()
        
        instances = sorted(instances, key=lambda(k,v): k)
        instanceCount = []
        instanceCount.append(len(instances))
        
        # Just to be sure!
        # remove with less than 2 (incoming) packets
        remove = []
        for outlierFeature, instance in instances:
            if len(instance.packets) <= 2 or outlierFeature < 2*512:
                #instances.remove((outlierFeature, instance))
                remove.append((outlierFeature, instance))
        instances = [x for x in instances if x not in remove]
        instanceCount.append(len(instances))
        
        # skip if number of instances is not sufficient
        if instanceCount[-1] < numOfInstances:
            print('WARN: ' + nameDir +'/' + filename + ' only ' + str(instanceCount[-1]) + '/' + str(instanceCount[0]) + ' of ' + str(numOfInstances))
            
            # Check if we have to discard everything
            if dataSet != '_':
                # Single file, so we have a problem! :(
                print('ERROR: ' + nameDir +'/' + filename + ' only ' + str(instanceCount[-1]) + '/' + str(instanceCount[0]) + ' of ' + str(numOfInstances))
                fdout.close()
                # Remove output for all formats
                for _, form in formats:
                    try:
                        os.remove(featurePath + dataSet + '_' + form)
                        os.remove(featurePath + 'list_' + dataSet[0].lower() + dataSet[1:] + '_' + form + '.txt')
                    except OSError as e:
                        if e.errno != errno.ENOENT:
                            raise
                break
            
            # We want synchronized instances, so we remove this page in every instance
            for outForm, _ in formats:
                try:
                    os.remove(featurePath + 'feature-' + outForm + '/' + filename)
                except OSError as e:
                    if e.errno != errno.ENOENT:
                        raise
            
            # We don't want to output this page
            continue
        
        # We have enough instances, so output name into info file
        if dataSet != '_':
            fdoutInfo.write(filename + '\n')
        
        print('INFO: (' + str(count) + '/' + str(len(inFiles)) + ') ' +  nameDir +'/' + filename) #+ ' - ' + str(len(instances)) #+ '\n \t' + ''.join(['%d/' % (instanceCount[i]) for i in range(1,len(instanceCount))]) + '/' + str(instanceCount[0]))
        
        # Choose instances randomly -> results in unsynchronized files
        if numOfInstances == len(instances):
            samples = instances
        else:
            if randomInstances:
                print('WARN: This is dangerous! Instances are probably not synchronized anymore!')
                samples = random.sample(instances, numOfInstances)
            else:
                samples = instances[:numOfInstances]
        
        # Calculate Features
        for _, instance in samples:
            features = []
            
            total = []
            cum = []
            pos = []
            neg = []
            inSize = 0
            outSize = 0
            inCount = 0
            outCount = 0
            
            # Process trace
            for item in itertools.islice(instance.packets, None): 
                packetsize = int(item.packetsize)
                
                # incoming packets
                if packetsize > 0:
                    inSize += packetsize
                    inCount += 1
                    # cumulated packetsizes
                    if len(cum) == 0:
                        cum.append(packetsize)
                        total.append(packetsize)
                        pos.append(packetsize)
                        neg.append(0)
                    else:
                        cum.append(cum[-1] + packetsize)
                        total.append(total[-1] + abs(packetsize))
                        pos.append(pos[-1] + packetsize)
                        neg.append(neg[-1] + 0)
            
                # outgoing packets
                if packetsize < 0:
                    outSize += abs(packetsize)
                    outCount += 1
                    if len(cum) == 0:
                        cum.append(packetsize)
                        total.append(abs(packetsize))
                        pos.append(0)
                        neg.append(abs(packetsize))
                    else:
                        cum.append(cum[-1] + packetsize)
                        total.append(total[-1] + abs(packetsize))
                        pos.append(pos[-1] + 0)
                        neg.append(neg[-1] + abs(packetsize))
            
            # Should already be removed by outlier Removal
            #if len(cum) < 2:
                # something must be wrong with this capture
                #continue
            
            # add feature
            features.append(classLabel)
            features.append(inCount)
            features.append(outCount)
            features.append(outSize)
            features.append(inSize)
            
            if separateClassifier:
                # cumulative in and out
                posFeatures = numpy.interp(numpy.linspace(total[0], total[-1], featureCount/2), total, pos)
                negFeatures = numpy.interp(numpy.linspace(total[0], total[-1], featureCount/2), total, neg)
                for el in itertools.islice(posFeatures, None):
                    features.append(el)
                for el in itertools.islice(negFeatures, None):
                    features.append(el)
            else:
                # cumulative in one
                cumFeatures = numpy.interp(numpy.linspace(total[0], total[-1], featureCount+1), total, cum)
                for el in itertools.islice(cumFeatures, 1, None):
                    features.append(el)
            
            fdout.write(str(features[0]) + ' '  + ' '.join(['%d:%s' % (i+1, el) for i,el in enumerate(features[1:])]) + ' # ' + str(instance.timestamp) + '\n')
        
        # We have to close the output file for each page
        if dataSet == '_':
            fdout.close()
        
        if setting == 'CW':
            classLabel += 1
        count +=1
        
    countList[current] = count
    current += 1
    
    if dataSet != '_':
        fdout.close()
        fdoutInfo.close()
