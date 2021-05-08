#!/bin/bash -l 

arg=${1:-j}
pkg=CSGOptiX
bin=CSGOptiXFlight

export CFBASE=/tmp/$USER/opticks/CSG_GGeo 
[ ! -d "$CFBASE/CSGFoundry" ] && echo ERROR no such directory $CFBASE/CSGFoundry && exit 1

render()
{
    local ridx=$1
    export OUTDIR=/tmp/$USER/opticks/$pkg/$bin/$(CSGOptiXVersion)/$ridx    # see CSGOptiX/tests/CSGOptiXVersion.cc
    mkdir -p $OUTDIR

    export LOGDIR=${OUTDIR}.logs
    mkdir -p $LOGDIR 
    cd $LOGDIR 

    export CSGOptiX=INFO
    export CUDA_VISIBLE_DEVICES=${CVD:-0}

    case $ridx in
      j) TMIN=1.5 EYE=-1.0,-1.0,1.0,1.0  CAMERATYPE=1 TOP=i0 $GDB $bin 0  ;;  
      i) TMIN=0.2 EYE=-0.5,0.0,0.0,1.0  CAMERATYPE=0 TOP=i0 $GDB $bin 0  ;;
      0) TMIN=0.4 EYE=-0.4,0.0,0.0,1.0  CAMERATYPE=1 $bin $ridx  ;;
      1) TMIN=0.5 EYE=-0.8,0.0,0.0,1.0  CAMERATYPE=1 $bin $ridx  ;;
      2) TMIN=0.5 EYE=-0.8,0.0,0.0,1.0  CAMERATYPE=1 $bin $ridx  ;;
      3) TMIN=0.5 EYE=-0.8,0.0,0.0,1.0  CAMERATYPE=1 $bin $ridx  ;;
      4) TMIN=0.5 EYE=-1.0,0.0,0.0,1.0  CAMERATYPE=1 $bin $ridx  ;;
      5) TMIN=0.5 EYE=-1.0,0.0,-0.2,1.0 CAMERATYPE=1 $bin $ridx  ;;
      6) TMIN=0.5 EYE=-1.0,0.0,-0.2,1.0 CAMERATYPE=1 $bin $ridx  ;;    ## Greek Temple : dias too small for columns
      7) TMIN=0.5 EYE=-1.0,0.0,0.0,1.0  CAMERATYPE=1 $bin $ridx  ;;
      8) TMIN=0.001 TMAX=1000. EYE=-2.0,-2.0,2.0,1.0  CAMERATYPE=1 $bin $ridx  ;;     ## giant crazy sphere prim, blank render 
      9) TMIN=0.7 EYE=-0.8,0.0,0.5,1.0  CAMERATYPE=1 $bin $ridx  ;;
      *) TMIN=0.5 EYE=-0.8,0.0,0.0,1.0  $bin $ridx  ;;
    esac
}


render $arg 



exit 0 
