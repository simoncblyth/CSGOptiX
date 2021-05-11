#!/bin/bash -l 

nmm=${NMM:-9}   # geometry specific 

usage(){ cat << EOU
::

    cxrscan.sh 
    NMM=5 cxrscan.sh 

EOU
}

scan-ee()
{
    echo "t0"
    for e in $(seq 0 $nmm) ; do echo  "$e," ; done
    for e in $(seq 0 $nmm) ; do echo "t$e," ; done
    for e in $(seq 0 $nmm) ; do echo "t8,$e" ; done
    echo "1,2,3,4"   # ONLY PMTs
}

for e in $(scan-ee) 
do 
    echo $e 
    EMM=$e cxr.sh $*
done 


