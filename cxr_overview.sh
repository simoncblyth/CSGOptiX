#!/bin/bash -l 

export MOI=-1 
export TMIN=0.4 
export EYE=-0.6,0,0,1 
export CAM=0 
export ZOOM=1.5 
export QUALITY=90 


export NAMEPREFIX=cxr_overview_${moi}_
export RELDIR=cxr_overview/cam_${CAM}

stamp=$(date +"%Y-%m-%d %H:%M")
version=$(CSGOptiXVersion)

export TOPLINE="./cxr_overview.sh    # EYE $EYE   $stamp  $version " 

source ./cxr.sh  

   

