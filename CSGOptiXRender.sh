#!/bin/bash -l 
usage(){ cat << EOU

::

   ./CSGOptiXRender.sh 

   TMIN is in units of extent, so when on axis disappearance at TMIN 2 is to be expected

EOU
}

#moi=sStrut:10:0
moi=sChimneySteel:0:0
moi=${1:-$moi}

pkg=CSGOptiX
bin=CSGOptiXRender

#export CSGOptiX=INFO
export CUDA_VISIBLE_DEVICES=${CVD:-0}

export CFBASE=/tmp/$USER/opticks/CSG_GGeo 
[ ! -d "$CFBASE/CSGFoundry" ] && echo ERROR no such directory $CFBASE/CSGFoundry && exit 1


render()
{
    export OUTDIR=/tmp/$USER/opticks/$pkg/$bin/$(CSGOptiXVersion)/$MOI
    mkdir -p $OUTDIR

    export LOGDIR=${OUTDIR}.logs
    mkdir -p $LOGDIR 
    cd $LOGDIR 

    $GDB $bin $* 

    jpg=$OUTDIR/pixels.jpg
    [ ! -f "$jpg" ] &&  echo FAILED TO CREATE jpg $jpg && exit 1

    echo $jpg && ls -l $jpg
    [ "$(uname)" == "Darwin" ] && open $jpg
}



if [ "$moi" == "ALL" ]; then 

    names=$(cat $CFBASE/CSGFoundry/name.txt | grep -v Flange | sort | uniq | perl -ne 'm,(.*0x).*, && print "$1\n" ' - )
    for name in $names ; do 
       echo $name 
       export MOI=$name 
       render 
    done

else
    export MOI=${MOI:-$moi}  
    render 
fi 


exit 0 
