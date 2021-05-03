#!/bin/bash -l 

msg="=== $BASH_SOURCE :"
sdir=$(pwd)
name=$(basename $sdir)

chkvar()
{
    local msg="=== $FUNCNAME :"
    local var ; 
    for var in $* ; do 
        if [ -z "${!var}" -o ! -d "${!var}" ]; then 
            echo $msg missing required envvar $var ${!var} OR non-existing directory
            return 1
        fi
        printf "%20s : %s \n" $var ${!var}
    done
    return 0  
} 

chkvar OPTICKS_PREFIX OPTICKS_HOME
[ $? -ne 0 ] && echo $msg checkvar FAIL && return 1

buildenv=$PREFIX/build/buildenv.sh
[ -f $buildenv ] && source $buildenv 

bdir=/tmp/$USER/opticks/$name/build

rm -rf $bdir && mkdir -p $bdir 
[ ! -d $bdir ] && exit 1
cd $bdir && pwd 

cmake $sdir \
     -DCMAKE_BUILD_TYPE=Debug \
     -DOPTICKS_PREFIX=${OPTICKS_PREFIX} \
     -DCMAKE_MODULE_PATH=${OPTICKS_HOME}/cmake/Modules \
     -DCMAKE_INSTALL_PREFIX=${OPTICKS_PREFIX}

[ $? -ne 0 ] && echo $msg conf FAIL && exit 1


make
[ $? -ne 0 ] && echo $msg make FAIL && exit 2

make install   
[ $? -ne 0 ] && echo $msg install FAIL && exit 3


exit 0

