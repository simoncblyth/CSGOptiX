#!/bin/bash -l 
usage(){ cat << EOU

::

   ./CSGOptiXRender.sh 

   TMIN is in units of extent, so when on axis disappearance at TMIN 2 is to be expected

   Formerly with ridx 8  

   TMIN=2.0 EYE=-1.0,0.0,0.0,1.0  CAMERATYPE=1 $bin $ridx   ## blank grey
   TMIN=1.99 EYE=-1.0,0.0,0.0,1.0  CAMERATYPE=1 $bin $ridx   ## pink square 

  j) TMIN=1.5 EYE=-1.0,-1.0,1.0,1.0  CAMERATYPE=1 TOP=i0 $GDB $bin $*  ;; 
  i) TMIN=0.2 EYE=-0.5,0.0,0.0,1.0  CAMERATYPE=0 TOP=i0 $GDB $bin $*  ;;
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

EOU
}

pkg=CSGOptiX
bin=CSGOptiXRender

export CFBASE=/tmp/$USER/opticks/CSG_GGeo 
[ ! -d "$CFBASE/CSGFoundry" ] && echo ERROR no such directory $CFBASE/CSGFoundry && exit 1

## get 0-based meshIdx from : cat.py GItemList/GMeshLib.txt 
## TODO: provide a name based approach, as the indices change meaning often
## 117: sChimneySteel0x4e6eff0

export MIDX=${MIDX:-117}  ## 
export OUTDIR=/tmp/$USER/opticks/$pkg/$bin/$(CSGOptiXVersion)/$MIDX
mkdir -p $OUTDIR

export LOGDIR=${OUTDIR}.logs
mkdir -p $LOGDIR 
cd $LOGDIR 

export CSGOptiX=INFO
export CUDA_VISIBLE_DEVICES=${CVD:-0}

$GDB $bin $* 

jpg=$OUTDIR/pixels.jpg
[ ! -f "$jpg" ] &&  echo FAILED TO CREATE jpg $jpg && exit 1

echo $jpg && ls -l $jpg
[ "$(uname)" == "Darwin" ] && open $jpg

exit 0 
