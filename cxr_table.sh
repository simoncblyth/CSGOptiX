#!/bin/bash -l 

basedir=/tmp/$USER/opticks/CSGOptiX/CSGOptiXRender/70000/render/CSG_GGeo/1/i0
reldir=-1

snap.py --basedir $basedir --reldir -1 --refjpgpfx /env/presentation/cxr/cxr_overview $*

