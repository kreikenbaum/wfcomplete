#! /bin/bash
### evaluates panchenkos code on thesis's data format, open world scenario

P_PATH=~/da/git/sw/p
SVM_PATH=$P_PATH/libsvm-3.20-src
#SVM_PATH=$P_PATH/../libsvm-3.22
D1="p_batch"
D2="or"
D3="features"

if [ $# -lt 2 ]; then
    echo "usage: p_open.sh FG_SCENARIO BG_SCENARIO"
    exit 1
fi
FG=$1
#echo $FG
BG=$2
#echo $BG

### background
cd $BG || exit 1
if ! [ -d $D1 ]; then
    $P_PATH/../../bin/panchenko_batch.py;
fi
if ! [ -d $D2 ]; then
    $P_PATH/feature-scripts/outlier-removal.py -in ./$D1/ -out ./$D2/ -setting CW -randomInstances NO -instances 1 -referenceFormat tcp -outlierRemoval Simple -ignoreOutlier NO;
fi
if ! [ -d $D3 ]; then
    python $P_PATH/feature-scripts/generate-feature.py -in ./$D2/ -out ./$D3/ -setting CW -classifier CUMULATIVE -force NO -features 100 -randomInstances NO -dataSet CW -instances 1
fi

### foreground
cd $FG || exit 2
if ! [ -d $D1 ]; then
    $P_PATH/../../bin/panchenko_batch.py;
fi
INSTANCES=$($P_PATH/../../bin/panchenko_instance_num.sh)
echo "$INSTANCES instances for each class" > $($P_PATH/../../bin/path_filename.sh p-cumul.out)
if ! [ -d $D2 ]; then
    python $P_PATH/feature-scripts/outlier-removal.py -in ./$D1/ -out ./$D2/ -setting CW -randomInstances NO -instances $INSTANCES -referenceFormat tcp -outlierRemoval Simple -ignoreOutlier NO;
fi
if ! [ -d $D3 ]; then
    python $P_PATH/feature-scripts/generate-feature.py -in ./$D2/ -out ./$D3/ -setting CW -classifier CUMULATIVE -force NO -features 100 -randomInstances NO -dataSet CW -instances $INSTANCES
fi

### combine
cat $D3/CW_TCP $BG/$D3/CW_TCP > $D3/OW_TCP

# last two might be better done by eval.py
$SVM_PATH/svm-scale -l 0 $D3/OW_TCP > $D3/OW_TCP_scaled
python $SVM_PATH/tools/grid.py -log2c 0,24,4 -log2g -15,5,4 -v 3 -gnuplot "null" ./$D3/OW_TCP_scaled >> $($P_PATH/../../bin/path_filename.sh p-cumul.out)
# nice $SVM_PATH/tools/grid.py -log2c 0,24,4 -log2g -15,5,4 -v 3 -gnuplot "null" ./output/CW_TCP_scaled >> $($P_PATH/../../bin/path_filename.sh p-cumul.out)
