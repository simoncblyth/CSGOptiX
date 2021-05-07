#include <sstream>

#include "SSys.hh"
#include "OPTICKS_LOG.hh"
#include "Opticks.hh"

#include "sutil_vec_math.h"
#include "CSGFoundry.h"
#include "CSGOptiX.h"
#include "CXUtil.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 

    Opticks ok(argc, argv); 
    ok.configure(); 

    const char* outdir = SSys::getenvvar("OUTDIR", "/tmp" ); 
    std::stringstream tt ; 
    tt << outdir << "/" << "pixels.jpg" ; 
    std::string path = tt.str(); 

    int repeatIdx = argc > 1 ? atoi(argv[1]) : 0 ; 

    std::stringstream uu ; 
    uu << "g" << repeatIdx ; 
    std::string s = uu.str(); 
    std::string top = CXUtil::GetEValue<std::string>("TOP", s.c_str() ); 

    std::string cfbase = CXUtil::GetEValue<std::string>("CFBASE", "/tmp" );  
    const char* rel = "CSGFoundry" ; 
    std::cout << " CSGFoundry::Load from " << cfbase << "/" << rel << std::endl ; 
    CSGFoundry* fd = CSGFoundry::Load(cfbase.c_str(), rel); // $CFBASE/CSGFoundry 
    fd->upload(); 
    std::cout << " fdd " << fd->desc() << std::endl ; 
    fd->summary(); 
    //fd->dump(); 


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
    cx.setTop( top.c_str() ); 

    cx.setCE(ce, tmin_model, tmax_model); 
    double dt = cx.render();  


    std::string top_line = "CSGOptiXRender" ; 
    std::stringstream ss ; 
    ss << std::fixed << std::setw(10) << std::setprecision(4) << dt ;  
    std::string bottom_line = ss.str(); 

    cx.snap(path.c_str(), bottom_line.c_str(), top_line.c_str() );   

    return 0 ; 
}
