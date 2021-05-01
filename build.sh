#!/bin/bash -l 

sdir=$(pwd)

source ./env.sh 

if [ -z "$PREFIX" -o ! -d "$PREFIX" ]; then 
   echo $0 PREFIX envvar not defined or directory $PREFIX does not exist 
   exit 1 
fi
if [ -z "$NAME" ]; then
   echo $0 NAME envvar not defined
   exit 2
fi


bdir=$PREFIX/build 
echo bdir $bdir NAME $NAME PREFIX $PREFIX

rm -rf $bdir && mkdir -p $bdir 
[ ! -d $bdir ] && exit 1
cd $bdir && pwd 


glm-version(){ echo 0.9.9.8 ; }
glm-name(){    echo glm-$(glm-version) ; }
glm-dir(){     echo $PREFIX/externals/glm/$(glm-name) ; }
glm-prefix(){  echo $(glm-dir)/glm ; }
glm-url(){     echo https://github.com/g-truc/glm/releases/download/$(glm-version)/$(glm-name).zip ; }
glm-dist(){    echo $(dirname $(glm-dir))/$(basename $(glm-url)) ; }
glm-get(){
   local msg="=== $FUNCNAME :"
   local iwd=$PWD
   local dir=$(dirname $(glm-dir)) &&  mkdir -p $dir && cd $dir
   local url=$(glm-url)
   local zip=$(basename $url)
   local nam=$(glm-name)
   local opt=$( [ -n "${VERBOSE}" ] && echo "" || echo "-q" )

   local hpp=$nam/glm/glm/glm.hpp
   echo $msg nam $nam PWD $PWD hpp $hpp
   ## curiously directories under /tmp being emptied but directory structure
   ## remains, so have to check file rather than directory existance  

   [ ! -f "$zip" ] && curl -L -O $url
   [ ! -f "$hpp" ] && unzip $opt $zip -d $nam
   ln -sfnv $(glm-name)/glm glm 
   echo symbolic link for access without version in path
   cd $iwd
}
glm-get

if [ "${GLM_PREFIX}" == "$(glm-prefix)" ]; then
   echo $BASH_SOURCE GLM_PREFIX $GLM_PREFIX is consistent with glm-prefix $(glm-prefix)
else
   echo $BASH_SOURCE GLM_PREFIX $GLM_PREFIX is NOT consistent with glm-prefix $(glm-prefix)
   exit 2 
fi
 

cmake $sdir \
     -DCMAKE_BUILD_TYPE=Debug \
     -DOptiX_INSTALL_DIR=${OPTIX_PREFIX} \
     -DCMAKE_MODULE_PATH=${OPTIX_PREFIX}/SDK/CMake \
     -DCMAKE_INSTALL_PREFIX=$PREFIX

rm -f $PREFIX/ptx/*.ptx
rm -f $PREFIX/bin/$NAME

make
[ $? -ne 0 ] && echo $0 : make FAIL && exit 1
make install   
[ $? -ne 0 ] && echo $0 : install FAIL && exit 2


exit 0

