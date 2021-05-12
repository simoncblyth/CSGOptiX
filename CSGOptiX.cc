#include <iostream>
#include <cstdlib>

#include <optix.h>
#if OPTIX_VERSION < 70000
#else
#include <optix_stubs.h>
#endif

#include <cuda_runtime.h>
#include <glm/glm.hpp>

#include "SStr.hh"
#include "SSys.hh"
#include "SMeta.hh"
#include "SVec.hh"

#include "BTimeStamp.hh"
#include "PLOG.hh"
#include "Opticks.hh"
#include "Composition.hh"
#include "FlightPath.hh"


#include "sutil_vec_math.h"

#include "CSGPrim.h"
#include "CSGFoundry.h"

#include "CSGView.h"
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


CSGOptiX::CSGOptiX(Opticks* ok_, const CSGFoundry* foundry_) 
    :
    ok(ok_),
    flight(ok->hasArg("--flightconfig")),
    width(ok->getWidth()),
    height(ok->getHeight()),
    depth(1),
    composition(ok->getComposition()),
    foundry(foundry_),
    prefix(ENV("OPTICKS_PREFIX","/usr/local/opticks")),
    outdir(ok->getOutDir()),    // evar:OUTDIR default overridden by --outdir option   
    cmaketarget("CSGOptiX"),  
    ptxpath(SStr::PTXPath( prefix, cmaketarget, PTXNAME )),
#if OPTIX_VERSION < 70000 
    geoptxpath(SStr::PTXPath(prefix, cmaketarget, GEO_PTXNAME )),
#else
    geoptxpath(nullptr),
#endif
    tmin_model(SSys::getenvfloat("TMIN",0.1)), 
    jpg_quality(SStr::GetEValue<int>("QUALITY", 50)),
    params(new Params(width, height, depth)),
#if OPTIX_VERSION < 70000
    six(new Six(ok, ptxpath, geoptxpath, params)),
#else
    ctx(new Ctx(params)),
    pip(new PIP(ptxpath)), 
    sbt(new SBT(ok, pip)),
    frame(new Frame(width, height)),  // CUDA holds the pixels 
#endif
    meta(new SMeta)
{
    init(); 
}

void CSGOptiX::init()
{
    LOG(LEVEL) << "[" ; 
    assert( prefix && "expecting PREFIX envvar pointing to writable directory" );
    assert( outdir && "expecting OUTDIR envvar " );

    LOG(LEVEL) << " ptxpath " << ptxpath  ; 
    LOG(LEVEL) << " geoptxpath " << ( geoptxpath ? geoptxpath : "-" ) ; 

    initGeometry();
    initFrame(); 

    LOG(LEVEL) << "]" ; 
}


void CSGOptiX::initGeometry()
{
    params->node = foundry->d_node ; 
    params->plan = foundry->d_plan ; 
    params->tran = nullptr ; 
    params->itra = foundry->d_itra ; 

    bool is_uploaded =  params->node != nullptr ;
    if(!is_uploaded) LOG(fatal) << "foundry must be uploaded prior to CSGOptiX::initGeometry " ;  
    assert( is_uploaded ); 

#if OPTIX_VERSION < 70000
    six->setFoundry(foundry);
#else
    sbt->setFoundry(foundry); 
#endif

}

void CSGOptiX::initFrame()
{
#if OPTIX_VERSION < 70000
    six->initFrame();     // sets params->pixels, isect from optix device pointers
#else
    params->pixels = frame->getDevicePixels(); 
    params->isect  = frame->getDeviceIsect(); 
#endif
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

/**
CSGOptiX::setCE
------------------

Setting center_extent establishes the coordinate system. 

**/

void CSGOptiX::setCE(const float4& v )
{
    glm::vec4 ce(v.x, v.y, v.z, v.w); 
    setCE(ce); 
}
void CSGOptiX::setCE(const glm::vec4& ce )
{
    bool aim = true ; 
    composition->setCenterExtent(ce, aim);  // model2world view setup 

    float extent = ce.w ; 
    float tmin = extent*tmin_model ; 
    LOG(info) 
        << " ce [ " << ce.x << " " << ce.y << " " << ce.z << " " << ce.w << "]" 
        << " tmin_model " << tmin_model
        << " tmin " << tmin 
        ; 

    composition->setNear(tmin); 
}


void CSGOptiX::updateView()
{
    float extent = composition->getExtent(); 

    glm::vec3 eye ;
    glm::vec3 U ; 
    glm::vec3 V ; 
    glm::vec3 W ; 
    glm::vec4 ZProj ;

    composition->getEyeUVW(eye, U, V, W, ZProj); // must setModelToWorld in composition first

    float tmin = composition->getNear(); 
    float tmax = composition->getFar(); 
    unsigned cameratype = composition->getCameraType(); 

    LOG(info)
        << " extent " << extent
        << " tmin " << tmin 
        << " tmax " << tmax 
        << " eye (" << eye.x << " " << eye.y << " " << eye.z << " ) "
        << " U (" << U.x << " " << U.y << " " << U.z << " ) "
        << " V (" << V.x << " " << V.y << " " << V.z << " ) "
        << " W (" << W.x << " " << W.y << " " << W.z << " ) "
        ;

    params->setView(eye, U, V, W);
    params->setCamera(tmin, tmax, cameratype ); 

#if OPTIX_VERSION < 70000
    six->updateContext(); 
#else
    ctx->uploadParams();  
#endif

    LOG(info)
        << "composition.desc " << std::endl 
        << composition->desc() 
        ; 

}

double CSGOptiX::render()
{
    updateView(); 

    LOG(LEVEL) << "[" ; 
    double t0, t1 ; 
    t0 = BTimeStamp::RealTime();
#if OPTIX_VERSION < 70000
    six->launch(); 
#else
    CUstream stream;
    CUDA_CHECK( cudaStreamCreate( &stream ) );
    OPTIX_CHECK( optixLaunch( pip->pipeline, stream, ctx->d_param, sizeof( Params ), &(sbt->sbt), frame->width, frame->height, frame->depth ) );
    CUDA_SYNC_CHECK();
#endif
    t1 = BTimeStamp::RealTime();
    double dt = t1 - t0 ; 
    frame_times.push_back(dt);  

    LOG(LEVEL) << "] " << std::fixed << std::setw(7) << std::setprecision(4) << dt  ; 
    return dt ; 
}


std::string CSGOptiX::Annotation( double dt )  // static 
{
    std::stringstream ss ; 
    ss << std::fixed << std::setw(10) << std::setprecision(4) << dt ;  
    std::string anno = ss.str(); 
    return anno ; 
}



void CSGOptiX::snap(const char* path, const char* bottom_line, const char* top_line, unsigned line_height)
{
#if OPTIX_VERSION < 70000
    six->snap(path, bottom_line, top_line, line_height); 
#else
    frame->download(); 
    frame->annotate( bottom_line, top_line, line_height ); 
    frame->writeJPG(path, jpg_quality);  
#endif
    if(!flight || SStr::Contains(path,"00000"))
    {
        saveMeta(path); 
    }
}


int CSGOptiX::render_flightpath()
{
    FlightPath* fp = ok->getFlightPath();   // FlightPath lazily instanciated here (held by Opticks)
    int rc = fp->render( (SRenderer*)this  );
    return rc ; 
}

void CSGOptiX::saveMeta(const char* jpg_path) const
{
    const char* json_path = SStr::ReplaceEnd(jpg_path, ".jpg", ".json"); 

    const std::vector<double>& t = frame_times ;
    double mn, mx, av ;
    SVec<double>::MinMaxAvg(t,mn,mx,av);

    const char* nameprefix = ok->getNamePrefix() ;

    nlohmann::json& js = meta->js ;
    js["argline"] = ok->getArgLine();
    js["nameprefix"] = nameprefix ; 
    js["jpg"] = jpg_path ; 
    js["emm"] = ok->getEnabledMergedMesh() ;
    js["mn"] = mn ;
    js["mx"] = mx ;
    js["av"] = av ;

    meta->save(json_path);
    LOG(info) << json_path ; 

    //const char* npy_path = SStr::ReplaceEnd(jpgpath, ".jpg", ".npy"); 
    //NP::Write(npy_path, (double*)t.data(),  t.size() );
}



