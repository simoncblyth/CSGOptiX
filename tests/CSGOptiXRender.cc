#include "SSys.hh"
#include "OPTICKS_LOG.hh"
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
    const char* top    = SSys::getenvvar("TOP", "i0" ); 
    const char* cfbase = SSys::getenvvar("CFBASE", "$TMP/CSG_GGeo" );

    CSGFoundry* fd = CSGFoundry::Load(cfbase, "CSGFoundry"); 
    fd->upload(); 
    LOG(info) << "foundry " << fd->desc() ; 
    fd->summary(); 

    const char* moi = SSys::getenvvar("MOI", "sWorld:0:0"); 
    int midx, mord, iidx ; 
    fd->parseMOI(midx, mord, iidx,  moi );  

    LOG(info) 
        << " MOI " << moi 
        << " midx " << midx 
        << " mord " << mord 
        << " iidx " << iidx
        ;   


    float4 ce = make_float4(0.f, 0.f, 0.f, 1000.f ); 
    int rc = fd->getCenterExtent(ce, midx, mord, iidx) ;
    if(rc != 0) return 1 ; 


    CSGOptiX cx(&ok, fd, outdir); 
    cx.setTop(top); 
    cx.setCE(ce);   // establish the coordinate system 


    bool flight = ok.hasArg("--flightconfig") ; 
    if(flight)
    {
        cx.render_flightpath() ; 
    }
    else
    {
        double dt = cx.render();  

        std::string path = CSGOptiX::Path(outdir, "pixels.jpg" );  
        std::string bottom_line = CSGOptiX::Annotation(dt); 
        std::string top_line = "CSGOptiXRender" ; 

        cx.snap(path.c_str(), bottom_line.c_str(), top_line.c_str() );   
    }
    return 0 ; 
}
