#!/bin/bash -l

usage(){ cat << EOU
::

    MOI=-1 EYE=-1,0,0,1 ./demo.sh 

EOU
}


cfname=CSGDemoTest
moi=0:0:4
eye=-10,0,5,1

export CFNAME=${CFNAME:-$cfname}
export MOI=${MOI:-$moi}

# change default eye when are targetting midx -1 (entire ias)
# TODO: pick between multiple IAS with midx -1,-2,...
[ "${MOI:0:2}" == "-1" ] && eye=-1,0,1,1   

export EYE=${EYE:-$eye}

./CSGOptiXRender.sh $*

