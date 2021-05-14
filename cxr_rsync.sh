#!/bin/bash -l 

usage(){ cat << EOU

From an environment with NAMEPREFIX defined run::

   source ./cxr_rsync.sh 

This will copy jpg with names starting with the NAMEPREFIX 
to the BASE directory with the source directory structure
beneath the TMPBASE dir preserved in the copy::

   TMPBASE  $TMPBASE

EOU
}


BASE=$HOME/simoncblyth.bitbucket.io/env/presentation
#BASE=/tmp/env/presentation
TMPBASE=/tmp/$USER/opticks/CSGOptiX/CSGOptiXRender

NAMEPREFIX=${NAMEPREFIX:-cxr_solid}
from=$TMPBASE
to=$BASE

vars="BASE TMPBASE NAMEPREFIX from to"
for var in $vars ; do printf " %20s : %s \n" $var ${!var} ; done 

rsync  -zarv --include="*/" --include="${NAMEPREFIX}*.jpg" --exclude="*" "$from" "$to"


find $BASE -name "${NAMEPREFIX}*.jpg"



