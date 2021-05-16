#!/bin/bash -l 

usage(){ cat << EOU
::

   EMM=0, ./cxr_overview.sh 
   EMM=1, ./cxr_overview.sh 



EOU
}


export MOI=-1 
export TMIN=0.4 
export EYE=-0.6,0,0,1 
export CAM=0 
export ZOOM=1.5 
export QUALITY=90 

[ "$(uname)" == "Darwin" ] && emm=1, || emm=t8,
export EMM=${EMM:-$emm}

export NAMEPREFIX=cxr_overview_${moi}_
export RELDIR=cxr_overview/cam_${CAM}_emm_${EMM}

stamp=$(date +"%Y-%m-%d %H:%M")
version=$(CSGOptiXVersion)

export TOPLINE="./cxr_overview.sh    # EYE $EYE MOI $MOI ZOOM $ZOOM   $stamp  $version " 
export BOTLINE=" RELDIR $RELDIR NAMEPREFIX $NAMEPREFIX  "

source ./cxr.sh  

exit 0    

