#!/bin/bash -l 

#moi=sStrut      # what to look at 
moi=sWaterTube   # should be same as lLowerChimney_phys

export MOI=${1:-$moi}
export TMIN=0.4 
export EYE=-1,-1,-1,1 
export CAM=0 
#export ZOOM=1.5 
export QUALITY=90 

export NAMEPREFIX=cxr_view_${moi}_
export RELDIR=cxr_view/cam_${CAM}

stamp=$(date +"%Y-%m-%d %H:%M")
version=$(CSGOptiXVersion)

export TOPLINE="./cxr_view.sh $MOI      # EYE $EYE   $stamp  $version " 

source ./cxr.sh     
#source cxr_rsync.sh 

