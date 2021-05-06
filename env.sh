#!/bin/bash 

msg="=== $BASH_SOURCE :"

sdir=$(pwd)
name=$(basename $sdir)

export CUDA_VISIBLE_DEVICES=0

#tmin=2.0
#tmin=1.5
#tmin=1.0
#tmin=0.5
tmin=0.1

tmax=100.

#geometry=parade
#geometry=sphere_containing_grid_of_spheres
#geometry=layered_sphere
#geometry=layered_zsphere
#geometry=clustered_sphere
#geometry=sphe
geometry=zsph
#geometry=cone
#geometry=hype
#geometry=box3 
#geometry=plan 
#geometry=slab  
#geometry=cyli
#geometry=disc 
#geometry=vcub
#geometry=vtet
#geometry=elli
#geometry=ubsp
#geometry=ibsp 
#geometry=dbsp
#geometry=rcyl


#clusterspec=-3:4:1,-3:4:1,-3:4:1
clusterspec=-1:2:1,-1:2:1,-1:2:1
clusterunit=500

gridmodulo=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14
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


#eye=-0.7,-0.7,0.15,1.0
eye=-0.7,-0.7,-0.1,1.0
#eye=-0.5,0.0,0.15,1.0
#eye=-0.5,-0.5,-0.5,1.0
#eye=-1.0,-1.0,0.0,1.0
#eye=-1.0,-1.0,1.0,1.0
#eye=-0.8,-0.8,0.8,1.0


cameratype=0
#cameratype=1

# number of concentric layers in compound shapes
layers=1     
#layers=2
#layers=3
#layers=20


export CSG_OPTIX_VERSION=$(CSGOptiXVersion)
export BINARY=CSGOptiXTest 
export OUTBASE=/tmp/$USER/opticks/CSGOptiX/$BINARY

# make sensitive to calling environment
export GEOMETRY=${GEOMETRY:-$geometry}
export TMIN=${TMIN:-$tmin}
export TMAX=${TMAX:-$tmax}
export CAMERATYPE=${CAMERATYPE:-$cameratype}

export CLUSTERSPEC=${CLUSTERSPEC:-$clusterspec}
export CLUSTERUNIT=${CLUSTERUNIT:-$clusterunit}

export GRIDSPEC=${GRIDSPEC:-$gridspec}
export GRIDMODULO=${GRIDMODULO:-$gridmodulo}
export GRIDSINGLE=${GRIDSINGLE:-$gridsingle}
export GRIDSCALE=${GRIDSCALE:-$gridscale}

export EYE=${EYE:-$eye} 
export LAYERS=${LAYERS:-$layers}
export OUTDIR=$OUTBASE/$GEOMETRY/${CSG_OPTIX_VERSION}

fmt="%-20s : %s \n"
printf "$fmt" name $name
printf "$fmt" CSG_OPTIX_VERSION ${CSG_OPTIX_VERSION}
printf "$fmt" BINARY $BINARY

printf "$fmt" GEOMETRY $GEOMETRY
printf "$fmt" TMIN $TMIN
printf "$fmt" TMAX $TMAX
printf "$fmt" CAMERATYPE $CAMERATYPE

printf "$fmt" GRIDSPEC $GRIDSPEC
printf "$fmt" GRIDMODULO $GRIDMODULO
printf "$fmt" GRIDSINGLE $GRIDSINGLE
printf "$fmt" GRIDSCALE $GRIDSCALE

printf "$fmt" EYE $EYE
printf "$fmt" LAYERS $LAYERS
printf "$fmt" OUTBASE $OUTBASE
printf "$fmt" OUTDIR $OUTDIR

return 0 
