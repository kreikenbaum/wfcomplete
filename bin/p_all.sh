#! /bin/sh
### evaluates panchenkos code on thesis's data format

P_PATH=~/da/git/sw/p
SVM_PATH=$P_PATH/libsvm-3.20-src
#SVM_PATH=$P_PATH/../libsvm-3.22
TARGET=p_batch

if [ $# -ge 1 ]; then
   TARGET=$1;
fi

if ! [ -d $TARGET ]; then
    $P_PATH/../../bin/panchenko_batch.py;
fi

INSTANCES=$($P_PATH/../../bin/panchenko_instance_num.sh)
echo "$INSTANCES instances for each class" > $($P_PATH/../../bin/path_filename.sh p-cumul.out)

python $P_PATH/feature-scripts/outlier-removal.py -in ./$TARGET/ -out ./or/ -setting CW -randomInstances NO -instances $INSTANCES -referenceFormat tcp -outlierRemoval Simple -ignoreOutlier NO
python $P_PATH/feature-scripts/generate-feature.py -in ./or/ -out ./features/ -setting CW -classifier CUMULATIVE -force NO -features 100 -randomInstances NO -dataSet CW -instances $INSTANCES
# last two might be better done by eval.py
$SVM_PATH/svm-scale -l 0 features/CW_TCP > features/CW_TCP_scaled
python $SVM_PATH/tools/grid.py -log2c 0,24,4 -log2g -15,5,4 -v 3 -gnuplot "null" ./features/CW_TCP_scaled >> $($P_PATH/../../bin/path_filename.sh p-cumul.out)
#nice $SVM_PATH/tools/grid.py -log2c 0,24,4 -log2g -15,5,4 -v 3 -gnuplot "null" ./output/CW_TCP_scaled >> $($P_PATH/../../bin/path_filename.sh p-cumul.out)
