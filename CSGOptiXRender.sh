#!/bin/bash -l 
usage(){ cat << EOU

::

   MOI=sStrut:10:0 EYE=-1,-1,1,1 TMIN=1.5 ./CSGOptiXRender.sh 
   MOI=sChimneySteel:0:0                  ./CSGOptiXRender.sh 
   MOI=sChimneySteel                      ./CSGOptiXRender.sh 
   CVD=1 MOI=sChimneySteel                ./CSGOptiXRender.sh 
   MOI=ALL                               ./CSGOptiXRender.sh    



   Demo Geometry created by::

      cd ~/CSG
      ./CSGDemoTest.sh  

   Unclear about what MOI is selecting on... 

      CFNAME=CSGDemoTest MOI=0:0:4 EYE=-10,0,5,1  ./CSGOptiXRender.sh

   TODO: add "meshnames" 
   TODO: provide way to pick the ias (overall geometry) center-extent via MOI 

 
   TMIN is in units of extent, so when on axis disappearance at TMIN 2 of a box is to be expected

EOU
}

msg="=== $BASH_SOURCE :"

export MOI=${MOI:-sStrut}
export CUDA_VISIBLE_DEVICES=${CVD:-0}

echo $msg MOI $MOI CUDA_VISIBLE_DEVICES ${CUDA_VISIBLE_DEVICES}

pkg=CSGOptiX
bin=CSGOptiXRender

export CFNAME=${CFNAME:-CSG_GGeo}
export CFBASE=/tmp/$USER/opticks/${CFNAME} 
[ ! -d "$CFBASE/CSGFoundry" ] && echo ERROR no such directory $CFBASE/CSGFoundry && exit 1

export OUTDIR=/tmp/$USER/opticks/$pkg/$bin/$(CSGOptiXVersion)/render/${CFNAME}/${CUDA_VISIBLE_DEVICES}
mkdir -p $OUTDIR

arglist=$OUTDIR/arglist.txt

export LOGDIR=${OUTDIR}.logs
mkdir -p $LOGDIR 
cd $LOGDIR 

render(){ $GDB $bin $* ; }


make_arglist()
{
    local arglist=$1
    local mname=$CFBASE/CSGFoundry/name.txt 
    ls -l $mname
    #cat $mname | grep -v Flange | grep -v _virtual | sort | uniq | perl -ne 'm,(.*0x).*, && print "$1\n" ' -  > $arglist
    cat $mname | grep -v Flange | grep -v _virtual | sort | uniq > $arglist
    ls -l $arglist && cat $arglist 
}


if [ "$MOI" == "ALL" ]; then 
    make_arglist $arglist 
    render --arglist $arglist $* 
else
    render $*
    jpg=$OUTDIR/${MOI}.jpg
    echo $msg jpg $jpg 
    ls -l $jpg
    [ "$(uname)" == "Darwin" ] && open $jpg
fi 

exit 0 
