P_PATH=~/da/git/sw/p

INST=$(wc -l $(find p_batch/ -type f | head -1) | cut -d ' ' -f 1)

LOWEST=$(python $P_PATH/feature-scripts/outlier-removal.py -in ./p_batch/ -out ./or/ -setting CW -randomInstances NO -instances $INST -referenceFormat tcp -outlierRemoval Simple -ignoreOutlier NO | grep WARN | cut -d ' ' -f 4 | cut -d '/' -f 1 | sort -n | head -1)

if [ x$LOWEST != x ]; then
    echo $LOWEST;
else
    echo $INST
fi

