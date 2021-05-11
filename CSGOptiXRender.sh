#!/bin/bash -l 
usage(){ cat << EOU

::

    CVD=1 ./CSGOptiXRender.sh -e t8,
    CVD=0 ./CSGOptiXRender.sh -e t8,

    MOI=sStrut:10:0 EYE=-1,-1,1,1 TMIN=1.5 ./CSGOptiXRender.sh 
    MOI=sChimneySteel:0:0                  ./CSGOptiXRender.sh 
    MOI=sChimneySteel                      ./CSGOptiXRender.sh 
    CVD=1 MOI=sChimneySteel                ./CSGOptiXRender.sh 


    MOI=ALL ./CSGOptiXRender.sh    
       
      # MOI=ALL creates an arglist file and uses the --arglist option 
      # to create a sequence of renders at the positions specified in the arglist  


    Demo "parade" geometry created by::

      cd ~/CSG
      ./CSGDemoTest.sh  

   Unclear about what MOI is selecting on... 

   Option "-e" emm bitset skipping now works in both 7 and pre-7::

      CFNAME=CSGDemoTest MOI=0:0:4 EYE=-10,0,5,1  ./CSGOptiXRender.sh -e ~4,5,6
      CFNAME=CSGDemoTest MOI=0:0:4 EYE=-10,0,5,1  ./CSGOptiXRender.sh -e ~1,2,3,4,5,6,7,8,9
      CFNAME=CSGDemoTest MOI=0:0:4 EYE=-10,0,5,1 GDB=lldb_  ./CSGOptiXRender.sh -- -e ~1,2,3,4,5,6,7,8,9 
      CFNAME=CSGDemoTest MOI=0:0:4 EYE=-10,0,5,1 GDB=gdb ./CSGOptiXRender.sh -- -e ~15,14

   TODO: add "meshnames" 
   TODO: provide way to pick the ias (overall geometry) center-extent via MOI 

 
   TMIN is in units of extent, so when on axis disappearance at TMIN 2 of a box is to be expected

EOU
}

msg="=== $BASH_SOURCE :"


moi=sStrut
emm=t8,
cvd=1

export CVD=${CVD:-$cvd}
export EMM=${EMM:-$emm}
export MOI=${MOI:-$moi}

nameprefix=cxr_${EMM}_

echo $msg CVD $CVD EMM $EMM MOI $MOI nameprefix $nameprefix

pkg=CSGOptiX
bin=CSGOptiXRender

export CFNAME=${CFNAME:-CSG_GGeo}
export CFBASE=/tmp/$USER/opticks/${CFNAME} 
[ ! -d "$CFBASE/CSGFoundry" ] && echo ERROR no such directory $CFBASE/CSGFoundry && exit 1

export OUTDIR=/tmp/$USER/opticks/$pkg/$bin/$(CSGOptiXVersion)/render/${CFNAME}/${CVD}
mkdir -p $OUTDIR

arglist=$OUTDIR/arglist.txt

export LOGDIR=${OUTDIR}.logs
mkdir -p $LOGDIR 
cd $LOGDIR 

render(){ $GDB $bin --nameprefix $nameprefix --cvd $CVD --e $EMM $* ; }   # MOI enters via arglist 


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
