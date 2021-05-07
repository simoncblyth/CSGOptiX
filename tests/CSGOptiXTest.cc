#include <string>
#include <sstream>

#include "SSys.hh"
#include "OPTICKS_LOG.hh"
#include "Opticks.hh"

#include "sutil_vec_math.h"
#include "CSGFoundry.h"
#include "CSGOptiX.h"
#include "CXUtil.h"

#include "DemoGeo.h"
#include "glm/glm.hpp"


int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    Opticks ok(argc, argv); 
    ok.configure(); 

    //Composition* composition = ok->getComposition(); 

    const char* outdir = SSys::getenvvar("OUTDIR", "/tmp" );  
    std::stringstream tt ; 
    tt << outdir << "/" << "pixels.jpg" ; 
    std::string path = tt.str(); 

    CSGFoundry foundry ; 
    DemoGeo dg(&foundry) ;  
    dg.write(outdir);  
    const float4 gce = dg.getCenterExtent() ;  
    glm::vec4 ce(gce.x,gce.y,gce.z, gce.w );   // defines the center-extent of the region to view

    foundry.dump(); 
    foundry.upload();   // uploads nodes, planes, transforms

    CSGOptiX cx(&ok, &foundry, outdir); 
    cx.setTop( dg.top );    // sets geometry handle to be traced

    //float tmin_model = CXUtil::GetEValue<float>("TMIN", 0.1) ;
    //float tmax_model = CXUtil::GetEValue<float>("TMAX", 100.0) ;

    cx.setCE(ce);       // thru to composition
    cx.updateView();   //  updates and uploads "constants" 
    double dt = cx.render(); 

    std::string top_line = "CSGOptiXTest" ; 
    std::stringstream ss ; 
    ss << std::fixed << std::setw(10) << std::setprecision(4) << dt ;  
    std::string bottom_line = ss.str(); 

    cx.snap(path.c_str(), bottom_line.c_str(), top_line.c_str() );   

    return 0 ; 
}
