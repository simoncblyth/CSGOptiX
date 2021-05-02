
#include "sutil_vec_math.h"
#include "CSGFoundry.h"
#include "CSGOptiX.h"
#include "DemoGeo.h"
#include "glm/glm.hpp"

int main(int argc, char** argv)
{
    CSGFoundry foundry ; 
    CSGOptiX cx(&foundry); 

    DemoGeo dg(&foundry) ;  
    dg.write(cx.outdir);  
    foundry.dump(); 
    foundry.upload();   // uploads nodes, planes, transforms

    const float4 gce = dg.getCenterExtent() ;  
    glm::vec4 ce(gce.x,gce.y,gce.z, gce.w*1.4f );   // defines the center-extent of the region to view

    cx.setCE(ce, dg.tmin, dg.tmax); 
    cx.render( dg.top ); 

    return 0 ; 
}
