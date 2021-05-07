#include <sstream>
#include "sutil_vec_math.h"

#include "SSys.hh"
#include "OPTICKS_LOG.hh"
#include "Opticks.hh"
#include "FlightPath.hh"


#include "CSGFoundry.h"
#include "CSGOptiX.h"
#include "CXUtil.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    Opticks ok(argc, argv); 
    ok.configure(); 

    const char* outdir = SSys::getenvvar("OUTDIR", "/tmp" ); 
    FlightPath* fp = ok.getFlightPath(); 


    int repeatIdx = 0 ; 
    std::string top = CXUtil::GetEValue<std::string>("TOP", "i0" ); 

    std::string cfbase = CXUtil::GetEValue<std::string>("CFBASE", "/tmp" );  
    const char* rel = "CSGFoundry" ; 
    std::cout << " CSGFoundry::Load from " << cfbase << "/" << rel << std::endl ; 
    CSGFoundry* fd = CSGFoundry::Load(cfbase.c_str(), rel); // $CFBASE/CSGFoundry 
    fd->upload(); 
    std::cout << " fdd " << fd->desc() << std::endl ; 
    fd->summary(); 


    const CSGSolid* so = fd->getSolid(repeatIdx); 
    float extent = so->center_extent.w ; 
    std::cout 
        << " repeatIdx " << repeatIdx 
        << " extent " << extent 
        << " so " << so->desc() 
        << std::endl 
        ; 

    const float4 gce = so->center_extent ; 
    glm::vec4 ce(gce.x,gce.y,gce.z, gce.w );  // defines the center-extent of the region to view

    float tmin_model = CXUtil::GetEValue<float>("TMIN", 0.1) ;
    float tmax_model = CXUtil::GetEValue<float>("TMAX", 100.0) ;

    CSGOptiX cx(&ok, fd, outdir); 
    cx.setTop(top.c_str());

    cx.setCE(ce, tmin_model, tmax_model); 
    cx.render();  

    return 0 ; 
}
