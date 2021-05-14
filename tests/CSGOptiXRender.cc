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
    const char* solid_label = ok.getSolidLabel(); 
    std::vector<unsigned>& solid_selection = ok.getSolidSelection(); 

    CSGFoundry* fd = CSGFoundry::Load(cfbase, "CSGFoundry"); 
    fd->upload(); 

    // hmm maybe eliminate one_gas_ias as solid_label more general 
    // because the string label is more friendly an input approach

    if( solid_label )
    {
        fd->findSolidIdx(solid_selection, solid_label); 
        LOG(error) 
            << " --solid_label " << solid_label
            << " solid_selection.size  " << solid_selection.size() 
            ;
    }
    unsigned num_select = solid_selection.size();  


    LOG(info) << "foundry " << fd->desc() ; 
    //fd->summary(); 

    CSGOptiX cx(&ok, fd); 
    cx.setTop(top); 

    bool flight = ok.hasArg("--flightconfig") ; 
    const std::vector<std::string>& arglist = ok.getArgList() ;  // --arglist /path/to/arglist.txt
    std::string top_line = SSys::getenvvar("TOPLINE", "CSGOptiXRender") ; 

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
        const char* namestem = num_select == 0 ? arg.c_str() : SSys::getenvvar("NAMESTEM", "")  ; 

        int rc = 0 ; 
        float4 ce = make_float4(0.f, 0.f, 0.f, 1000.f ); 
        if(solid_selection.size() > 0)
        {
            unsigned target = solid_selection.size()/2 ;  // target the middle selected solid
            fd->gasCE(ce, solid_selection[target] );  
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
                const char* ext = ".jpg" ; 
                int index = -1 ;  
                const char* path = ok.getOutPath(namestem, ext, index ); 
                LOG(error) << " path " << path ; 

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
