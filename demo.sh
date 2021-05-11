#!/bin/bash -l

usage(){ cat << EOU
::

    MOI=-1 EYE=-1,0,0,1 ./demo.sh    ## MOI=-1 targets CE of entire IAS i0  

    MOI=0:0:4           ./demo.sh -e t4,5,6
    MOI=0:0:4           ./demo.sh -e t1,2,3,4,5,6,7,8,9
    MOI=0:0:4 GDB=lldb_ ./demo.sh -- -e t1,2,3,4,5,6,7,8,9 
    MOI=0:0:4 GDB=gdb   ./demo.sh -- -e t15,14

TODO: add "meshnames" to demo for name based MOI targetting 
TODO: pick between multiple IAS with midx -1,-2,...

The EYE, LOOK, UP envvars set the okc/View::home defaults 

EOU
}

cfname=CSGDemoTest    # picks the CSGFoundry geometry to load
moi=0:0:4             # what to look at 
eye=-10,0,5,1         # where to look from 

export CFNAME=${CFNAME:-$cfname}
export MOI=${MOI:-$moi}

[ "${MOI:0:2}" == "-1" ] && eye=-1,0,1,1  # change default eye when are targetting midx -1 (entire ias)

export EYE=${EYE:-$eye}

./cxr.sh $*

