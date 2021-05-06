
#include "sutil_vec_math.h"
#include "CSGFoundry.h"
#include "CSGOptiX.h"
#include "CXUtil.h"

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

    float tmin_model = CXUtil::GetEValue<float>("TMIN", 0.1) ;
    float tmax_model = CXUtil::GetEValue<float>("TMAX", 100.0) ;

    cx.setCE(ce, tmin_model, tmax_model ); 
    cx.render( dg.top ); 

    return 0 ; 
}
