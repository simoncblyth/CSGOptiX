#!/bin/bash -l 

usage(){ cat << EOU

::

   ./CSGOptiXRender.sh 0
   ./CSGOptiXRender.sh 1
   ./CSGOptiXRender.sh 2
   ..

   ./CSGOptiXRender.sh 8   ## bizarre, usually blank but now get a box

   TMIN is in units of extent, so when on axis disappearance at TMIN 2 is to be expected

   TMIN=2.0 EYE=-1.0,0.0,0.0,1.0  CAMERATYPE=1 $bin $ridx   ## blank grey
   TMIN=1.99 EYE=-1.0,0.0,0.0,1.0  CAMERATYPE=1 $bin $ridx   ## pink square 


EOU
}

arg=$1
pkg=CSGOptiX
bin=CSGOptiXRender

export CFBASE=/tmp/$USER/opticks/CSG_GGeo 
[ ! -d "$CFBASE/CSGFoundry" ] && echo ERROR no such directory $CFBASE/CSGFoundry && exit 1


render()
{
    local ridx=$1

    export OUTDIR=/tmp/$USER/opticks/$pkg/$bin/$(CSGOptiXVersion)/$ridx    # see CSGOptiX/tests/CSGOptiXVersion.cc
    mkdir -p $OUTDIR
    export CUDA_VISIBLE_DEVICES=0

    case $ridx in
      i) TOP=i0 CAMERATYPE=1 $bin 0  ;;
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

    jpg=$OUTDIR/pixels.jpg
    if [ ! -f "$jpg" ]; then 
       echo FAIL TO CREATE jpg $jpg && return 1
    fi  

    echo $jpg
    ls -l $jpg

    if [ "$(uname)" == "Darwin" ]; then
       open $jpg
    fi
    return 0
}

if [ "$arg" == "a" ]; then 
    for r in $(seq 0 9)
    do  
        render $r
        [ $? -ne 0 ] && echo FAIL for r $r && exit 1
    done
else
    render $arg
    [ $? -ne 0 ] && echo FAIL for arg $arg && exit 2
fi 

exit 0 
