P_PATH=~/da/git/sw/p

if ! [ -d p_batch ]; then
    $P_PATH/../../bin/panchenko_batch.py;
fi

INSTANCES=$($P_PATH/../../bin/panchenko_instance_num.sh)
echo "$INSTANCES instances for each class" > $($P_PATH/../../bin/path_filename.sh p-cumul.out)

python $P_PATH/feature-scripts/outlier-removal.py -in ./p_batch/ -out ./or/ -setting CW -randomInstances NO -instances $INSTANCES -referenceFormat tcp -outlierRemoval Simple -ignoreOutlier NO
python $P_PATH/feature-scripts/generate-feature.py -in ./or/ -out ./output/ -setting CW -classifier CUMULATIVE -force NO -features 100 -randomInstances NO -dataSet CW -instances $INSTANCES
# last two might be better done by eval.py
$P_PATH/libsvm-3.20-src/svm-scale -l 0 output/CW_TCP > output/CW_TCP_scaled  
nice $P_PATH/libsvm-3.20-src/tools/grid.py -log2c 0,24,4 -log2g -15,5,4 -v 3 ./output/CW_TCP_scaled >> $($P_PATH/../../bin/path_filename.sh p-cumul.out)