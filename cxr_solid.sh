#!/bin/bash -l 

usage(){ cat << EOU
cxr_solid.sh
============

Create render showing the Prim within a solid translated 
in Y for visibility.  

::

    ./cxr_solid.sh r1p
    ./cxr_solid.sh r2p
    ./cxr_solid.sh r3p
    ./cxr_solid.sh r4p

    ./cxr_solid.sh R2P0N

    ./cxr_solid.sh R8P0N11,R8P0N13,R8P0N14,R8P0N16,R8P0N17,R8P0N18,R8P0N19,R8P0N20,R8P0N21,R8P0N22,R8P0N15,R8P0N26


EOU
}


sla=${1:-r1p}

export SLA=$sla
export CAM=1
export EYE=0,-5,0,1 
export LOOK=0,0,0,1
#export GDB=lldb_ 

export NAMEPREFIX=cxr_solid_${sla}_
export RELDIR=cxr_solid/cam_${CAM}

stamp=$(date +"%Y-%m-%d %H:%M")
version=$(CSGOptiXVersion)

export TOPLINE="./cxr_solid.sh $SLA      # EYE $EYE   $stamp  $version " 

./cxr.sh  


