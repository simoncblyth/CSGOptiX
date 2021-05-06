#!/bin/bash -l

geometrys=$(perl -ne 'm,geometry=(\S*), && print "$1\n"' env.sh)

for geometry in $geometrys ; do 
    GEOMETRY=$geometry ./run.sh   
    [ $? -ne 0 ] && echo FAIL for geometry $geometry && exit 1
done

exit 0
