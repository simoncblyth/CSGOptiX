#!/bin/bash 

export CSG_PREFIX=${CSG_PREFIX:-/usr/local/csg}

if [ -z "$OPTIX_PREFIX" ]; then 
    echo $0 : missing required envvar OPTIX_PREFIX 
    exit 1 
fi 
if [ ! -d "$OPTIX_PREFIX"  ]; then 
    echo $0 : OPTIX_PREFIX $OPTIX_PREFIX directory does not exist
    exit 2
fi 
if [ -z "$CSG_PREFIX" ]; then 
    echo $0 : missing required envvar CSG_PREFIX 
    exit 1 
fi 
if [ ! -d "$CSG_PREFIX"  ]; then 
    echo $0 : CSG_PREFIX $CSG_PREFIX directory does not exist
    exit 2
fi 


sdir=$(pwd)
name=$(basename $sdir)

export PREFIX=/tmp/$USER/opticks/$name
buildenv=$PREFIX/build/buildenv.sh
if [ -f $buildenv ]; then 
    source $buildenv 
fi

case $(uname) in
  Darwin) var=DYLD_LIBRARY_PATH ; lib="lib" ;;
  Linux)  var=LD_LIBRARY_PATH ; lib="lib64"  ;;  
esac

export PATH=$PREFIX/bin:$PATH
export $var=$PREFIX/$lib:${CSG_PREFIX}/$lib:${OPTIX_PREFIX}/lib64
export BIN=$(which $name 2>/dev/null)

printf "$var\n${!var}" | tr ":" "\n" 



#tmin=2.0
#tmin=1.5
#tmin=1.0
#tmin=0.5
tmin=0.1

geometry=parade
#geometry=sphere_containing_grid_of_spheres
#geometry=layered_sphere
#geometry=layered_zsphere
#geometry=clustered_sphere
#geometry=sphe # 0
#geometry=zsph # 1 
#geometry=cone # 2
#geometry=hype # 3
#geometry=box3 # 4 
#geometry=plan # 5 
#geometry=slab # 6  
#geometry=cyli # 7
#geometry=disc # 8 
#geometry=vcub # 9
#geometry=vtet # 10
#geometry=elli # 11
#geometry=ubsp # 12 
#geometry=ibsp # 13 
#geometry=dbsp # 14
#geometry=rcyl  # 15



#clusterspec=-3:4:1,-3:4:1,-3:4:1
clusterspec=-1:2:1,-1:2:1,-1:2:1
clusterunit=500

gridmodulo=0,1,2
#gridmodulo=12,13,14
#gridmodulo=9,10
#gridmodulo=5,6
#gridmodulo=10
#gridmodulo=2
#gridsingle=2
gridsingle=""

#gridspec=-10:11:2,-10:11:2,-10:11:2
#gridspec=-10:11:2,-10:11:2,0:8:2
gridspec=-10:11:2,-10:11:2,0:6:3
#gridspec=-40:41:4,-40:41:4,-40:41:4
#gridspec=-40:41:10,-40:41:10,-40:41:10
#gridspec=-40:41:10,-40:41:10,0:1:1

gridscale=200.0


eye=-0.7,-0.7,0.15,1.0
#eye=-0.5,0.0,0.15,1.0
#eye=-0.5,-0.5,-0.5,1.0
#eye=-1.0,-1.0,0.0,1.0
#eye=-1.0,-1.0,1.0,1.0
#eye=-0.8,-0.8,0.8,1.0



cameratype=0
#cameratype=1

# number of concentric layers in compound shapes
#layers=1     
#layers=2
#layers=3
layers=20


snap=/tmp/${geometry}_zsphere_grid.sh 
echo writing snap $snap
cat << EOB > $snap
# $snap
tmin=$tmin
geometry=$geometry
gridmodulo=$gridmodulo
gridsingle=$gridsingle
gridspec=$gridspec
gridscale=$gridscale
eye=$eye
cameratype=$cameratype
layers=$layers
EOB


# make sensitive to calling environment
export GEOMETRY=${GEOMETRY:-$geometry}
export TMIN=${TMIN:-$tmin}
export CAMERATYPE=${CAMERATYPE:-$cameratype}

export CLUSTERSPEC=${CLUSTERSPEC:-$clusterspec}
export CLUSTERUNIT=${CLUSTERUNIT:-$clusterunit}

export GRIDSPEC=${GRIDSPEC:-$gridspec}
export GRIDMODULO=${GRIDMODULO:-$gridmodulo}
export GRIDSINGLE=${GRIDSINGLE:-$gridsingle}
export GRIDSCALE=${GRIDSCALE:-$gridscale}

export EYE=${EYE:-$eye} 
export LAYERS=${LAYERS:-$layers}
export OUTDIR=$PREFIX/$GEOMETRY/TMIN_${TMIN}

fmt="%-20s : %s \n"
printf "$fmt" name $name
printf "$fmt" PREFIX $PREFIX
printf "$fmt" OPTIX_VERSION $OPTIX_VERSION
printf "$fmt" BIN $BIN

printf "$fmt" GEOMETRY $GEOMETRY
printf "$fmt" TMIN $TMIN
printf "$fmt" CAMERATYPE $CAMERATYPE

printf "$fmt" GRIDSPEC $GRIDSPEC
printf "$fmt" GRIDMODULO $GRIDMODULO
printf "$fmt" GRIDSINGLE $GRIDSINGLE
printf "$fmt" GRIDSCALE $GRIDSCALE

printf "$fmt" EYE $EYE
printf "$fmt" LAYERS $LAYERS
printf "$fmt" OUTDIR $OUTDIR

