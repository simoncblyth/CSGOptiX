
#include <sstream>
#include "sutil_vec_math.h"
#include "CSGFoundry.h"
#include "CSGOptiX.h"
#include "CXUtil.h"

int main(int argc, char** argv)
{
    int repeatIdx = argc > 1 ? atoi(argv[1]) : 0 ; 

    std::stringstream ss ; 
    ss << "g" << repeatIdx ; 
    std::string s = ss.str(); 
    std::string top = CXUtil::GetEValue<std::string>("TOP", s.c_str() ); 


    const char* dir = "/tmp" ; 
    const char* rel = "CSGOptiXGGeo" ; 
    CSGFoundry* fd = CSGFoundry::Load(dir, rel); 
    fd->upload(); 

    std::cout 
        << " fdd " << fd->desc() 
        << std::endl 
        ;

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
    glm::vec4 ce(gce.x,gce.y,gce.z, gce.w*1.4f );   // defines the center-extent of the region to view


    CSGOptiX cx(fd); 

    float tmin = CXUtil::GetEValue<float>("TMIN", 0.1) ;
    float tmax = CXUtil::GetEValue<float>("TMAX", 100.0) ;

    cx.setCE(ce, tmin, tmax); 
    cx.render( top.c_str() );  

    return 0 ; 
}
