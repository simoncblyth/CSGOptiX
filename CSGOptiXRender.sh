#!/bin/bash -l 

usage(){ cat << EOU

::

   ./CSGOptiXRender.sh 0
   ./CSGOptiXRender.sh 1
   ./CSGOptiXRender.sh 2
   ..

EOU
}

ridx=$1
export OUTDIR=/tmp/$USER/opticks/CSGOptiXRender/$(CSGOptiXVersion)/$ridx    # see CSGOptiX/tests/CSGOptiXVersion.cc
mkdir -p $OUTDIR
export CUDA_VISIBLE_DEVICES=0

bin=CSGOptiXRender

case $ridx in 
  0) TMIN=0.4 EYE=-0.4,0.0,0.0,1.0  CAMERATYPE=1 $bin $ridx  ;;
  1) TMIN=0.5 EYE=-0.8,0.0,0.0,1.0  CAMERATYPE=1 $bin $ridx  ;;
  2) TMIN=0.5 EYE=-0.8,0.0,0.0,1.0  CAMERATYPE=1 $bin $ridx  ;;
  3) TMIN=0.5 EYE=-0.8,0.0,0.0,1.0  CAMERATYPE=1 $bin $ridx  ;;
  4) TMIN=0.5 EYE=-1.0,0.0,0.0,1.0  CAMERATYPE=1 $bin $ridx  ;;
  5) TMIN=0.5 EYE=-1.0,0.0,-0.2,1.0 CAMERATYPE=1 $bin $ridx  ;;
  6) TMIN=0.5 EYE=-1.0,0.0,-0.2,1.0 CAMERATYPE=1 $bin $ridx  ;;    ## Greek Temple : dias too small for columns
  7) TMIN=0.5 EYE=-1.0,0.0,0.0,1.0  CAMERATYPE=1 $bin $ridx  ;;
  8) TMIN=0.5 EYE=-1.0,0.0,0.0,1.0  CAMERATYPE=1 $bin $ridx  ;;     ## giant crazy sphere prim, blank render 
  9) TMIN=0.7 EYE=-0.8,0.0,0.5,1.0  CAMERATYPE=1 $bin $ridx  ;;
  *) TMIN=0.5 EYE=-0.8,0.0,0.0,1.0  $bin $ridx  ;;
esac

jpg=$OUTDIR/pixels.jpg
echo $jpg
ls -l $jpg

if [ "$(uname)" == "Darwin" ]; then
   open $jpg
fi

exit 0 
