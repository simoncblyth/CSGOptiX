/**
CSGOptiXRender
=================

With option --arglist /path/to/arglist.txt each line of the arglist file 
is taken as an MOI specifying the center_extent box to target. 
Without an --arglist option the MOI envvar or default value  "sWorld:0:0" 
is consulted to set the target box.
 

**/

#include <algorithm>
#include <iterator>

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

    const char* top    = SSys::getenvvar("TOP", "i0" ); 
    const char* cfbase = SSys::getenvvar("CFBASE", "$TMP/CSG_GGeo" );

    int one_gas_ias = ok.getOneGASIAS(); 


    CSGFoundry* fd = CSGFoundry::Load(cfbase, "CSGFoundry"); 
    fd->upload(); 
    LOG(info) << "foundry " << fd->desc() ; 
    //fd->summary(); 

    CSGOptiX cx(&ok, fd); 
    cx.setTop(top); 

    bool flight = ok.hasArg("--flightconfig") ; 
    const std::vector<std::string>& arglist = ok.getArgList() ;  // --arglist /path/to/arglist.txt
    std::string top_line = "CSGOptiXRender" ; 

    std::vector<std::string> args ; 
    if( arglist.size() > 0 )
    {    
        std::copy(arglist.begin(), arglist.end(), std::back_inserter(args));
    }
    else
    {
        args.push_back(SSys::getenvvar("MOI", "sWorld:0:0"));  
    }

    for(unsigned i=0 ; i < args.size() ; i++)
    {
        const std::string& arg = args[i];

        int rc = 0 ; 
        float4 ce = make_float4(0.f, 0.f, 0.f, 1000.f ); 
        if(one_gas_ias > -1)
        {
            fd->gasCE(ce, one_gas_ias ); 
        }
        else
        {
            int midx, mord, iidx ; 
            fd->parseMOI(midx, mord, iidx,  arg.c_str() );  
            LOG(info) << " i " << i << " arg " << arg << " midx " << midx << " mord " << mord << " iidx " << iidx ;   
            rc = fd->getCenterExtent(ce, midx, mord, iidx) ;
        }

        if(rc == 0 )
        {
            cx.setCE(ce);   // establish the coordinate system 

            if(flight)
            {
                cx.render_flightpath(); 
            }
            else
            {
                double dt = cx.render();  

                const char* namestem = arg.c_str() ; 
                const char* ext = ".jpg" ; 
                int index = -1 ;  
                const char* path = ok.getOutPath(namestem, ext, index ); 

                std::string bottom_line = CSGOptiX::Annotation(dt); 
                cx.snap(path, bottom_line.c_str(), top_line.c_str() );   
            }
        }
        else
        {
            LOG(error) << " SKIPPING as failed to lookup CE " << arg ; 
        }
    }
    return 0 ; 
}
