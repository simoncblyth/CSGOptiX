#!/bin/bash -l

usage(){ cat << EOU
::

    ./cxr_demo.sh -1    ## MOI=-1 targets CE of entire IAS i0  

    EYE=0,0,1,1 ./cxr_demo.sh -1 

The EYE, LOOK, UP envvars set the okc/View::home defaults 

Sometimes necessary to rerun creation of the Demo geometry 
to get this to work after model changes::

   cd ~/CSG
   ./CSGDemoTest.sh 


TODO: add "meshnames" to demo for name based MOI targetting 
TODO: pick between multiple IAS with midx -1,-2,...


EOU
}

cfname=CSGDemoTest    # picks the CSGFoundry geometry to load
moi=0:0:4             # what to look at 
eye=-10,0,5,1         # where to look from 

emm=t0                # default to no solid skips with demo geometry 

export MOI=${1:-$moi}
export CFNAME=${CFNAME:-$cfname}
export EMM=${EMM:-$emm}

[ "${MOI:0:2}" == "-1" ] && eye=-1,0,1,1  # change default eye when are targetting midx -1 (entire ias)

export EYE=${EYE:-$eye}


export NAMEPREFIX=cxr_demo_      # MOI is appended by tests/CSGOptiXRender.cc when --solid_label yields no solids
export RELDIR=cxr_demo/cam_${CAM}

stamp=$(date +"%Y-%m-%d %H:%M")
version=$(CSGOptiXVersion)

export TOPLINE="./cxr_demo.sh $MOI      # EYE $EYE  $stamp  $version " 



./cxr.sh $*

