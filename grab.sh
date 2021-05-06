#!/bin/bash -l 

pkg=CSGOptiX
from=P:/tmp/$USER/opticks/$pkg/
to=/tmp/$USER/opticks/$pkg/

echo from $from
echo to $to

rsync -zarv --progress --include="*/" --include="*.jpg" --exclude="*" "$from" "$to"


#find ${to:0:-1} -name '*.jpg'    ## curious this works in commandline but gives error in script : -1: substring expression < 0
find ${to%/} -name '*.jpg'

