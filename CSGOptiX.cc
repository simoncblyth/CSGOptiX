#include <iostream>
#include <cstdlib>

#include <optix.h>
#if OPTIX_VERSION < 70000
#else
#include <optix_stubs.h>
#endif

#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "sutil_vec_math.h"

#include "CSGPrim.h"
#include "CSGFoundry.h"

#include "Util.h"
#include "View.h"
#include "Frame.h"
#include "Params.h"

#if OPTIX_VERSION < 70000
#include "Six.h"
#else
#include "Ctx.h"
#include "CUDA_CHECK.h"   
#include "OPTIX_CHECK.h"   
#include "PIP.h"
#include "SBT.h"
#endif

#include "CSGOptiX.h"

#if OPTIX_VERSION < 70000 
const char* CSGOptiX::PTXNAME = "OptiX6Test" ; 
const char* CSGOptiX::GEO_PTXNAME = "geo_OptiX6Test" ; 

#else
const char* CSGOptiX::PTXNAME = "OptiX7Test" ; 
const char* CSGOptiX::GEO_PTXNAME = nullptr ; 
#endif


const char* CSGOptiX::ENV(const char* key, const char* fallback)
{
    const char* value = getenv(key) ; 
    return value ? value : fallback ; 
}

CSGOptiX::CSGOptiX(const CSGFoundry* foundry_ ) 
    :
    foundry(foundry_),
    prefix(ENV("OPTICKS_PREFIX","/usr/local/opticks")),
    outdir(ENV("OUTDIR", "/tmp")),
    cmaketarget("CSGOptiX"),  
    ptxpath(Util::PTXPath( prefix, cmaketarget, PTXNAME )),
#if OPTIX_VERSION < 70000 
    geoptxpath(Util::PTXPath(prefix, cmaketarget, GEO_PTXNAME )),
#else
    geoptxpath(nullptr),
#endif
    cameratype(Util::GetEValue<unsigned>("CAMERATYPE", 0u )),
    jpg_quality(Util::GetEValue<int>("QUALITY", 50)),
    eye_model(-1.f, -1.f, 1.f, 1.f ),
    view(new View),
    params(new Params)
{
    init(); 
}

void CSGOptiX::init()
{
    assert( prefix && "expecting PREFIX envvar pointing to writable directory" );
    assert( outdir && "expecting OUTDIR envvar " );

    Util::GetEVec(eye_model, "EYE", "-1.0,-1.0,1.0,1.0"); 
    std::cout << " ptxpath " << ptxpath << std::endl ; 
    std::cout << " geoptxpath " << ( geoptxpath ? geoptxpath : "-" ) << std::endl ; 
}

void CSGOptiX::setCE(const glm::vec4& ce, float tmin, float tmax  )
{
    view->update(eye_model, ce, width, height) ; 
    view->dump(); 
    view->save(outdir); 

    params->setView(view->eye, view->U, view->V, view->W, tmin, tmax, cameratype ); 
    params->setSize(width, height, depth); 
}

void CSGOptiX::render(const char* tspec)
{
    params->node = foundry->d_node ; 
    params->plan = foundry->d_plan ; 
    params->tran = foundry->d_tran ; 
    params->itra = foundry->d_itra ; 

#if OPTIX_VERSION < 70000
    Six six(ptxpath, geoptxpath, params); 
    six.setFoundry(foundry);
    six.setTop(tspec); 
    six.launch(); 
    six.save(outdir); 
#else
    Ctx ctx(params) ;
    PIP pip(ptxpath); 
    SBT sbt(&pip);
    sbt.setFoundry(foundry); 
    sbt.setTop(tspec);

    AS* top = sbt.getTop(); 
    params.handle = top->handle ; 

    Frame frame(params->width, params->height, params->depth); 
    params->pixels = frame.getDevicePixels(); 
    params->isect  = frame.getDeviceIsect(); 
    ctx.uploadParams();  

    CUstream stream;
    CUDA_CHECK( cudaStreamCreate( &stream ) );
    OPTIX_CHECK( optixLaunch( pip.pipeline, stream, ctx.d_param, sizeof( Params ), &(sbt.sbt), frame.width, frame.height, frame.depth ) );
    CUDA_SYNC_CHECK();

    frame.download(); 
    frame.write(outdir, jpg_quality);  
#endif
}


