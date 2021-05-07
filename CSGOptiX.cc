#include <iostream>
#include <cstdlib>

#include <optix.h>
#if OPTIX_VERSION < 70000
#else
#include <optix_stubs.h>
#endif

#include <cuda_runtime.h>
#include <glm/glm.hpp>


#include "PLOG.hh"


#include "sutil_vec_math.h"

#include "CSGPrim.h"
#include "CSGFoundry.h"

#include "CXUtil.h"
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


CSGOptiX::CSGOptiX(const CSGFoundry* foundry_, const char* outdir_) 
    :
    foundry(foundry_),
    prefix(ENV("OPTICKS_PREFIX","/usr/local/opticks")),
    outdir(outdir_),
    cmaketarget("CSGOptiX"),  
    ptxpath(CXUtil::PTXPath( prefix, cmaketarget, PTXNAME )),
#if OPTIX_VERSION < 70000 
    geoptxpath(CXUtil::PTXPath(prefix, cmaketarget, GEO_PTXNAME )),
#else
    geoptxpath(nullptr),
#endif
    cameratype(CXUtil::GetEValue<unsigned>("CAMERATYPE", 0u )),
    jpg_quality(CXUtil::GetEValue<int>("QUALITY", 50)),
    eye_model(-1.f, -1.f, 1.f, 1.f ),
    view(new View),
    params(new Params),
#if OPTIX_VERSION < 70000
    six(new Six(ptxpath, geoptxpath, params)),
#else
    ctx(new Ctx(params)),
    pip(new PIP(ptxpath)), 
    sbt(new SBT(pip)),
#endif
    frame(new Frame(width, height, depth))
{
    init(); 
}

void CSGOptiX::init()
{
    std::cout << "[ CSGOptiX::init " << std::endl ; 
    assert( prefix && "expecting PREFIX envvar pointing to writable directory" );
    assert( outdir && "expecting OUTDIR envvar " );

    CXUtil::GetEVec(eye_model, "EYE", "-1.0,-1.0,1.0,1.0"); 
    std::cout << " ptxpath " << ptxpath << std::endl ; 
    std::cout << " geoptxpath " << ( geoptxpath ? geoptxpath : "-" ) << std::endl ; 
    std::cout << "] CSGOptiX::init " << std::endl ; 

    params->node = foundry->d_node ; 
    params->plan = foundry->d_plan ; 
    params->tran = foundry->d_tran ; 
    params->itra = foundry->d_itra ; 

    bool is_uploaded =  params->node != nullptr ;
    if(!is_uploaded) LOG(fatal) << "foundry must be uploaded prior to CSGOptiX::init " ;  
    assert( is_uploaded ); 

    std::cout << "[ CSGOptiX::init.setFoundry " << std::endl ; 
#if OPTIX_VERSION < 70000
    six->setFoundry(foundry);
#else
    sbt->setFoundry(foundry); 
    params->pixels = frame->getDevicePixels(); 
    params->isect  = frame->getDeviceIsect(); 
#endif
    std::cout << "] CSGOptiX::init.setFoundry " << std::endl ; 
}

void CSGOptiX::setTop(const char* tspec)
{
    std::cout 
        << "[ CSGOptiX::setTop " 
        << " tspec " << tspec 
        << std::endl 
        ; 

#if OPTIX_VERSION < 70000
    six->setTop(tspec); 
#else
    sbt->setTop(tspec);
    AS* top = sbt->getTop(); 
    params->handle = top->handle ; 
#endif
}

void CSGOptiX::setCE(const glm::vec4& ce, float tmin_model, float tmax_model )
{
    float extent = ce.w ; 
    float tmin = extent*tmin_model ; 
    float tmax = extent*tmax_model ; 

    std::cout 
        << "[ CSGOptiX::setCE " 
        << " extent " << extent
        << " tmin_model " << tmin_model 
        << " tmax_model " << tmax_model 
        << " tmin " << tmin 
        << " tmax " << tmax 
        << std::endl 
        ; 

    view->update(eye_model, ce, width, height) ; 
    view->dump(); 
    view->save(outdir); 

    params->setView(view->eye, view->U, view->V, view->W, tmin, tmax, cameratype ); 
    params->setSize(frame->width, frame->height, frame->depth); 

    std::cout << "] CSGOptiX::setCE " << std::endl ; 
}

void CSGOptiX::render()
{
    std::cout << "[ CSGOptiX::render " << std::endl ; 
#if OPTIX_VERSION < 70000
    six->launch(); 
    six->save(outdir); 
#else
    ctx->uploadParams();  

    CUstream stream;
    CUDA_CHECK( cudaStreamCreate( &stream ) );
    OPTIX_CHECK( optixLaunch( pip->pipeline, stream, ctx->d_param, sizeof( Params ), &(sbt->sbt), frame->width, frame->height, frame->depth ) );
    CUDA_SYNC_CHECK();

    frame->download(); 
    frame->write(outdir, jpg_quality);  
#endif
    std::cout << "] CSGOptiX::render " << std::endl ; 
}


void CSGOptiX::render_flightpath()
{

}




