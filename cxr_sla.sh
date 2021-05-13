#!/bin/bash -l 

usage(){ cat << EOU
cxr_sla.sh
============

Create render showing the Prim within a solid translated 
in Y for visibility.  

::

    ./cxr_sla.sh r1p
    ./cxr_sla.sh r2p
    ./cxr_sla.sh r3p
    ./cxr_sla.sh r4p

EOU
}


sla=${1:-r1p}

export SLA=$sla
export CAM=1
export EYE=0,-5,0,1 
export LOOK=0,0,0,1
#export GDB=lldb_ 

./cxr.sh  



