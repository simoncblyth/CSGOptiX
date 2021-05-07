#include <iostream>
#include <cstdlib>

#include <optix.h>
#if OPTIX_VERSION < 70000
#else
#include <optix_stubs.h>
#endif

#include <cuda_runtime.h>
#include <glm/glm.hpp>


#include "BTimeStamp.hh"
#include "PLOG.hh"
#include "Opticks.hh"
#include "FlightPath.hh"


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


const plog::Severity CSGOptiX::LEVEL = PLOG::EnvLevel("CSGOptiX", "DEBUG" ); 

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


CSGOptiX::CSGOptiX(Opticks* ok_, const CSGFoundry* foundry_, const char* outdir_) 
    :
    ok(ok_),
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
    frame(new Frame(width, height))
{
    init(); 
}

void CSGOptiX::init()
{
    LOG(LEVEL) << "[" ; 
    assert( prefix && "expecting PREFIX envvar pointing to writable directory" );
    assert( outdir && "expecting OUTDIR envvar " );

    CXUtil::GetEVec(eye_model, "EYE", "-1.0,-1.0,1.0,1.0"); 
    LOG(LEVEL) << " ptxpath " << ptxpath  ; 
    LOG(LEVEL) << " geoptxpath " << ( geoptxpath ? geoptxpath : "-" ) ; 

    params->node = foundry->d_node ; 
    params->plan = foundry->d_plan ; 
    params->tran = foundry->d_tran ; 
    params->itra = foundry->d_itra ; 

    bool is_uploaded =  params->node != nullptr ;
    if(!is_uploaded) LOG(fatal) << "foundry must be uploaded prior to CSGOptiX::init " ;  
    assert( is_uploaded ); 

#if OPTIX_VERSION < 70000
    six->setFoundry(foundry);
#else
    sbt->setFoundry(foundry); 
    params->pixels = frame->getDevicePixels(); 
    params->isect  = frame->getDeviceIsect(); 
#endif
    LOG(LEVEL) << "]" ; 
}

void CSGOptiX::setTop(const char* tspec)
{
    LOG(LEVEL) << "[" << " tspec " << tspec ; 

#if OPTIX_VERSION < 70000
    six->setTop(tspec); 
#else
    sbt->setTop(tspec);
    AS* top = sbt->getTop(); 
    params->handle = top->handle ; 
#endif

    LOG(LEVEL) << "]" << " tspec " << tspec ; 
}

void CSGOptiX::setCE(const glm::vec4& ce, float tmin_model, float tmax_model )
{
    float extent = ce.w ; 
    float tmin = extent*tmin_model ; 
    float tmax = extent*tmax_model ; 

    LOG(LEVEL)
        << "[" 
        << " extent " << extent
        << " tmin_model " << tmin_model 
        << " tmax_model " << tmax_model 
        << " tmin " << tmin 
        << " tmax " << tmax 
        ; 

    view->update(eye_model, ce, width, height) ; 
    view->dump(); 
    view->save(outdir); 

    params->setView(view->eye, view->U, view->V, view->W, tmin, tmax, cameratype ); 
    params->setSize(frame->width, frame->height, frame->depth); 
#if OPTIX_VERSION < 70000
#else
    ctx->uploadParams();  
#endif

    LOG(LEVEL) << "[" ; 
 }

double CSGOptiX::render()
{
    LOG(LEVEL) << "[" ; 
    double t0, t1 ; 
#if OPTIX_VERSION < 70000
    t0 = BTimeStamp::RealTime();
    six->launch(); 
    t1 = BTimeStamp::RealTime();
    six->save(outdir); 
#else

    t0 = BTimeStamp::RealTime();

    CUstream stream;
    CUDA_CHECK( cudaStreamCreate( &stream ) );
    OPTIX_CHECK( optixLaunch( pip->pipeline, stream, ctx->d_param, sizeof( Params ), &(sbt->sbt), frame->width, frame->height, frame->depth ) );
    CUDA_SYNC_CHECK();

    t1 = BTimeStamp::RealTime();

#endif
    double dt = t1 - t0 ; 
    LOG(LEVEL) << "] " << std::fixed << std::setw(7) << std::setprecision(4) << dt  ; 
    return dt ; 
}


void CSGOptiX::snap(const char* path, const char* bottom_line, const char* top_line, unsigned line_height)
{
    frame->download(); 
    frame->annotate( bottom_line, top_line, line_height ); 
    frame->writeJPG(path, jpg_quality);  
}


int CSGOptiX::render_flightpath()
{
    FlightPath* fp = ok->getFlightPath();   // FlightPath lazily instanciated here (held by Opticks)
    int rc = fp->render( (SRenderer*)this  );
    return rc ; 
}




