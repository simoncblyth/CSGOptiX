#!/bin/bash -l 


from=P:/tmp/$USER/opticks/CSGOptiXRender/ 
to=/tmp/$USER/opticks/CSGOptiXRender/

echo from $from
echo to $to

rsync -zarv --progress --include="*/" --include="*.jpg" --exclude="*" "$from" "$to"



