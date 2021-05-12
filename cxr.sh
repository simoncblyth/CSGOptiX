#!/bin/bash -l 
usage(){ cat << EOU

::


    OGI=0 ./cxr.sh     # one_gas_ias
    OGI=1 ./cxr.sh

    CVD=1 ./cxr.sh -e t8,
    CVD=0 ./cxr.sh -e t8,

    MOI=sStrut:10:0 EYE=-1,-1,1,1 TMIN=1.5 ./cxr.sh 
    MOI=sChimneySteel:0:0                  ./cxr.sh 
    MOI=sChimneySteel                      ./cxr.sh 
    CVD=1 MOI=sChimneySteel                ./cxr.sh 


    MOI=ALL ./cxr.sh    
       
      # MOI=ALL creates an arglist file and uses the --arglist option 
      # to create a sequence of renders at the positions specified in the arglist  


    Demo "parade" geometry created by::

      cd ~/CSG
      ./CSGDemoTest.sh  

   Unclear about what MOI is selecting on... 

   Option "-e" emm bitset skipping now works in both 7 and pre-7::

 
   TMIN is in units of extent, so when on axis disappearance at TMIN 2 of a box is to be expected


EYE, LOOK, UP envvars are read in okc/View::home : they define Composition view defaults.::

    281 void View::home()
    282 {
    ...
    288     m_eye = SGLM::EVec3("EYE", "-1,-1,0" );
    293     m_look = SGLM::EVec3("LOOK", "0,0,0" );
    298     m_up = SGLM::EVec3("UP", "0,0,1" );
    299 }




EOU
}

msg="=== $BASH_SOURCE :"


export CFNAME=${CFNAME:-CSG_GGeo}
export CFBASE=/tmp/$USER/opticks/${CFNAME} 
[ ! -d "$CFBASE/CSGFoundry" ] && echo ERROR no such directory $CFBASE/CSGFoundry && exit 1


pkg=CSGOptiX
bin=CSGOptiXRender

# defaults 
cvd=1            # default GPU to use
emm=t8,          # what to include in the GPU geometry 
#moi=sStrut      # what to look at 
moi=sWaterTube   # should be same as lLowerChimney_phys
eye=-1,-1,-1,1   # where to look from, see okc/View::home 
top=i0           # hmm difficuly to support other than i0
ogi=-1           # one_gas_ias less disruptive perhaps that changing top, -1 means disabled 
cam=0            # 0:perpective 1:orthographic 2:equirect (2:not supported in CSGOptiX(7) yet)
tmin=0.1         # near in units of extent, so typical range is 0.1-2.0 for visibility, depending on EYE->LOOK distance

[ "$(uname)" == "Darwin" ] && cvd=0    # only one GPU on laptop 

export CVD=${CVD:-$cvd}    # --cvd 
export EMM=${EMM:-$emm}    # -e 
export MOI=${MOI:-$moi}    # evar:MOI OR --arglist when MOI=ALL  
export EYE=${EYE:-$eye}    # evar:EYE 
export TOP=${TOP:-$top}    # evar:TOP?
export OGI=${OGI:-$ogi}    # --one_gas_ias
export CAM=${CAM:-$cam}    # evar:CAMERATYPE
export TMIN=${TMIN:-$tmin} # evar:TMIN

vars="CVD EMM MOI EYE TOP OGI CAM"
for var in $vars ; do printf "%10s : %s \n" $var ${!var} ; done 

export CAMERATYPE=$CAM    # okc/Camera::Camera default 


nameprefix=cxr_${top}_${EMM}_

export NAMEPREFIX=$nameprefix
export OUTDIR=/tmp/$USER/opticks/$pkg/$bin/$(CSGOptiXVersion)/render/${CFNAME}/${CVD}/${TOP}/${OGI}
mkdir -p $OUTDIR

arglist=$OUTDIR/arglist.txt

export LOGDIR=${OUTDIR}.logs
mkdir -p $LOGDIR 
cd $LOGDIR 


render-cmd(){ cat << EOC
$GDB $bin --nameprefix $nameprefix --cvd $CVD -e $EMM --one_gas_ias $OGI  $* 
EOC
}   

render()
{
   local msg="=== $FUNCNAME :"
   which $bin
   pwd

   local log=$bin.log
   local cmd=$(render-cmd $*) 
   echo $cmd

   printf "\n\n\n$cmd\n\n\n" >> $log 

   eval $cmd
   local rc=$?

   printf "\n\n\nRC $rc\n\n\n" >> $log 

   echo $msg rc $rc
}


make_arglist()
{
    local arglist=$1
    local mname=$CFBASE/CSGFoundry/name.txt  # /tmp/$USER/opticks/CSG_GGeo/CSGFoundry/name.txt  # mesh names
    ls -l $mname
    #cat $mname | grep -v Flange | grep -v _virtual | sort | uniq | perl -ne 'm,(.*0x).*, && print "$1\n" ' -  > $arglist
    cat $mname | grep -v Flange | grep -v _virtual | sort | uniq > $arglist
    ls -l $arglist && cat $arglist 
}


if [ "$MOI" == "ALL" ]; then 
    make_arglist $arglist 
    render --arglist $arglist $*            ## multiple MOI via the arglist 
else
    render $*                               ## single MOI via envvar 
    jpg=$OUTDIR/${nameprefix}${MOI}.jpg
    echo $msg jpg $jpg 
    ls -l $jpg
    [ "$(uname)" == "Darwin" ] && open $jpg
fi 

exit 0 
