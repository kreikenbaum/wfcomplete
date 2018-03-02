#! /bin/sh
### evaluates panchenkos code on thesis's data format, open world scenario

P_PATH=~/da/git/sw/p
SVM_PATH=$P_PATH/libsvm-3.20-src
#SVM_PATH=$P_PATH/../libsvm-3.22

if [ $# -lt 2 ]; then
    echo "usage: p_open.sh FG_SCENARIO BG_SCENARIO"
    exit 1
fi
FG=$1
BG=$2

### background
cd $BG || (echo "$BG not found"; exit 1)
if ! [ -d p_batch ]; then
    $P_PATH/../../bin/panchenko_batch.py;
fi
python $P_PATH/feature-scripts/outlier-removal.py -in ./p_batch/ -out ./or/ -setting CW -randomInstances NO -instances 1 -referenceFormat tcp -outlierRemoval Simple -ignoreOutlier NO
python $P_PATH/feature-scripts/generate-feature.py -in ./or/ -out ./features/ -setting CW -classifier CUMULATIVE -force NO -features 100 -randomInstances NO -dataSet CW -instances 1

### foreground
cd $FG || (echo "$FG not found"; exit 1)
if ! [ -d p_batch ]; then
    $P_PATH/../../bin/panchenko_batch.py;
fi
INSTANCES=$($P_PATH/../../bin/panchenko_instance_num.sh)
echo "$INSTANCES instances for each class" > $($P_PATH/../../bin/path_filename.sh p-cumul.out)
python $P_PATH/feature-scripts/outlier-removal.py -in ./p_batch/ -out ./or/ -setting CW -randomInstances NO -instances $INSTANCES -referenceFormat tcp -outlierRemoval Simple -ignoreOutlier NO
python $P_PATH/feature-scripts/generate-feature.py -in ./or/ -out ./features/ -setting CW -classifier CUMULATIVE -force NO -features 100 -randomInstances NO -dataSet CW -instances $INSTANCES

### combine
cat features/CW_TCP $BG/features/CW_TCP > features/OW_TCP

# last two might be better done by eval.py
$SVM_PATH/svm-scale -l 0 features/OW_TCP > features/OW_TCP_scaled
python $SVM_PATH/tools/grid.py -log2c 0,24,4 -log2g -15,5,4 -v 3 -gnuplot "null" ./output/OW_TCP_scaled >> $($P_PATH/../../bin/path_filename.sh p-cumul.out)
#nice $SVM_PATH/tools/grid.py -log2c 0,24,4 -log2g -15,5,4 -v 3 -gnuplot "null" ./output/CW_TCP_scaled >> $($P_PATH/../../bin/path_filename.sh p-cumul.out)
