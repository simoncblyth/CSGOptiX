#!/bin/bash -l 
usage(){ cat << EOU

::

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
    283     m_changed = true ;
    284 
    285     m_eye.x = -1.f ;
    286     m_eye.y = -1.f ;
    287     m_eye.z =  0.f ;
    288     m_eye = SGLM::EVec3("EYE", "-1,-1,0" );
    289 
    290     m_look.x =  0.f ;
    291     m_look.y =  0.f ;
    292     m_look.z =  0.f ;
    293     m_look = SGLM::EVec3("LOOK", "0,0,0" );
    294 
    295     m_up.x =  0.f ;
    296     m_up.y =  0.f ;
    297     m_up.z =  1.f ;
    298     m_up = SGLM::EVec3("UP", "0,0,1" );
    299 }




EOU
}

msg="=== $BASH_SOURCE :"

# defaults 
cvd=1            # which GPU to use
emm=t8,          # what to include in the GPU geometry 
#moi=sStrut      # what to look at 
moi=sWaterTube   # should be same as lLowerChimney_phys
eye=-1,-1,-1,1   # where to look from, see okc/View::home 
top=i0

[ "$(uname)" == "Darwin" ] && cvd=0 

export CVD=${CVD:-$cvd}
export EMM=${EMM:-$emm}
export MOI=${MOI:-$moi}
export EYE=${EYE:-$eye}
export TOP=${TOP:-$top}

nameprefix=cxr_${top}_${EMM}_

echo $msg CVD $CVD TOP $top EMM $EMM MOI $MOI EYE $EYE nameprefix $nameprefix

pkg=CSGOptiX
bin=CSGOptiXRender

export CFNAME=${CFNAME:-CSG_GGeo}
export CFBASE=/tmp/$USER/opticks/${CFNAME} 
[ ! -d "$CFBASE/CSGFoundry" ] && echo ERROR no such directory $CFBASE/CSGFoundry && exit 1

export OUTDIR=/tmp/$USER/opticks/$pkg/$bin/$(CSGOptiXVersion)/render/${CFNAME}/${CVD}/${TOP}
mkdir -p $OUTDIR

arglist=$OUTDIR/arglist.txt

export LOGDIR=${OUTDIR}.logs
mkdir -p $LOGDIR 
cd $LOGDIR 

render(){ $GDB $bin --nameprefix $nameprefix --cvd $CVD -e $EMM $* ; }   # MOI enters via arglist 


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
