#!/bin/bash -l

usage(){ cat << EOU
flight7.sh
===============

See also flight.sh 


    CVD=1 ./flight7.sh -e ~8,

Developments here need to follow cxr.sh to some extent. 

EOU
}

msg="=== $0 :"

export CFNAME=${CFNAME:-CSG_GGeo}
export CFBASE=/tmp/$USER/opticks/${CFNAME} 
[ ! -d "$CFBASE/CSGFoundry" ] && echo $msg ERROR no such directory $CFBASE/CSGFoundry && exit 1

pkg=CSGOptiX
bin=CSGOptiXRender

period=${PERIOD:-16}
limit=${LIMIT:-1024}
scale0=${SCALE0:-3}
scale1=${SCALE1:-0.5}
flight=${FLIGHT:-RoundaboutXY_XZ}


# defaults 
cvd=1            # which GPU to use
emm=t8,          # what to include in the GPU geometry 
#moi=sStrut      # what to look at 
moi=sWaterTube   # sWaterTube should be same as lLowerChimney_phys
eye=-1,-1,-1,1   # where to look from, see okc/View::home 
top=i0           # hmm difficuly to support
ogi=-1           # one_gas_ias less disruptive perhaps : -1 default means not enabled
cam=0            # 0:perpective 1:orthographic 2:equirect (not supported in CSGOptiX(7) yet)

[ "$(uname)" == "Darwin" ] && cvd=0 

export CVD=${CVD:-$cvd}
export EMM=${EMM:-$emm}
export MOI=${MOI:-$moi}
export EYE=${EYE:-$eye}
export TOP=${TOP:-$top}
export OGI=${OGI:-$ogi}
export CAM=${CAM:-$cam}

export CAMERATYPE=$CAM     # okc/Camera default 

vars="CVD EMM MOI EYE TOP OGI CAM"
for var in $vars ; do printf "%10s : %s \n" $var ${!var} ; done 

nameprefix=cxr_${top}_${EMM}_
outbase=/tmp/$USER/opticks/$pkg/$bin/$(CSGOptiXVersion)/flight/$CFNAME/$CVD/$TOP/$OGI

prefix="${flight}__${MOI}"
outdir="$outbase/$prefix"
config="flight=$flight,ext=.jpg,scale0=$scale0,scale1=$scale1,framelimit=$limit,period=$period"
mkdir -p $outdir 

flight-cmd(){ cat << EOC
$bin --flightconfig "$config" --flightoutdir "$outdir" --nameprefix "$prefix" --cvd $CVD -e $EMM --one_gas_ias $OGI  $*
EOC
}

flight-render-jpg()
{
   local msg="=== $FUNCNAME :"
   which $bin
   pwd

   echo $msg creating output directory outdir: "$outdir"
   mkdir -p "$outdir" 

   local log=$bin.log
   local cmd=$(flight-cmd $*) 
   echo $cmd

   printf "\n\n\n$cmd\n\n\n" >> $log 
   eval $cmd
   local rc=$?
   printf "\n\n\nRC $rc\n\n\n" >> $log 

   echo $msg rc $rc
}

flight-make-mp4()
{
    local msg="=== $FUNCNAME :"
    local jpg2mp4=$HOME/env/bin/ffmpeg_jpg_to_mp4.sh
    [ ! -x "$jpg2mp4" ] && echo $msg no jpg2mp4 $jpg2mp4 script && return 1 

    cd "$outdir" 
    pwd
    $jpg2mp4 "$prefix"

    return 0 
}

flight-render()
{
    flight-render-jpg $*
    flight-make-mp4
}

flight-grab()
{
    [ -z "$outbase" ] && echo $msg outbase $outbase not defined && return 1 
    local cmd="rsync -rtz --progress P:$outbase/ $outbase/"
    echo $cmd
    eval $cmd
    open $outbase
    return 0 
}

if [ "$(uname)" == "Darwin" ]; then
    flight-grab
else
    flight-render $*
fi 

