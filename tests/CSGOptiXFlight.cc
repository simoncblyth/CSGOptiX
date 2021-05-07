#include "OPTICKS_LOG.hh"
#include "SSys.hh"
#include "Opticks.hh"

#include "sutil_vec_math.h"
#include "CSGFoundry.h"
#include "CSGOptiX.h"

int main(int argc, char** argv)
{
    OPTICKS_LOG(argc, argv); 
    Opticks ok(argc, argv); 
    ok.configure(); 

    const char* outdir = SSys::getenvvar("OUTDIR", "/tmp" ); 
    const char* top = SSys::getenvvar("TOP", "i0" ); 
    const char* cfbase = SSys::getenvvar("CFBASE", "/tmp" );  

    CSGFoundry* foundry = CSGFoundry::Load(cfbase, "CSGFoundry"); 
    foundry->upload(); 

    LOG(info) << "foundry " << foundry->desc() ; 
    foundry->summary(); 

    int repeatIdx = 0 ; 
    const CSGSolid* so = foundry->getSolid(repeatIdx); 
    const float4 ce = so->center_extent ; 

    LOG(info) << " repeatIdx " << repeatIdx << " ce.w " << ce.w << " so " << so->desc() ; 

    CSGOptiX cx(&ok, foundry, outdir); 
    cx.setTop(top);
    cx.setCE(ce); 

    double dt = cx.render();  

    std::string path = CSGOptiX::Path(outdir, "pixels.jpg" );  
    std::string bottom_line = CSGOptiX::Annotation( dt ); 
    std::string top_line = "CSGOptiXFlight" ; 

    cx.snap(path.c_str(), bottom_line.c_str(), top_line.c_str() );   

    return 0 ; 
}
