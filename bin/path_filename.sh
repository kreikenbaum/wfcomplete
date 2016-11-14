# generate filename from directory
if [ $# -eq 0 ]; then
    echo needs name as first parameter;
    exit
fi

if [ -d /tmp/mem ]; then
    RESULT=/tmp/mem/
else
    RESULT=./
fi
echo $RESULT$(echo $(pwd)/$1 | sed 's!/!___!g')
