
#include <iostream>

#include "PLOG.hh"
#include "Params.h"

#include "sutil_vec_math.h"
#include "OpticksCSG.h"
#include "CSGFoundry.h"
#include "CSGSolid.h"
#include "CSGPrim.h"
#include "CSGNode.h"

#include "Six.h"

    
Six::Six(const char* ptx_path_, const char* geo_ptx_path_, Params* params_)
    :
    context(optix::Context::create()),
    material(context->createMaterial()),
    params(params_),
    ptx_path(strdup(ptx_path_)),
    geo_ptx_path(strdup(geo_ptx_path_)),
    entry_point_index(0u),
    optix_device_ordinal(0u),
    foundry(nullptr)
{
    initContext();
    initPipeline(); 
    initFrame(); 
    updateContext(); 
}

void Six::initContext()
{
    LOG(info); 
    context->setRayTypeCount(1);
    context->setPrintEnabled(true);
    context->setPrintBufferSize(40960);
    context->setPrintLaunchIndex(-1,-1,-1); 
    context->setEntryPointCount(1);
}

void Six::initPipeline()
{
    LOG(info); 
    context->setRayGenerationProgram( entry_point_index, context->createProgramFromPTXFile( ptx_path , "raygen" ));
    context->setMissProgram(   entry_point_index, context->createProgramFromPTXFile( ptx_path , "miss" ));

    material->setClosestHitProgram( entry_point_index, context->createProgramFromPTXFile( ptx_path, "closest_hit" ));
}

void Six::initFrame()
{
    LOG(info); 
    pixels_buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4, params->width, params->height);
    context["pixels_buffer"]->set( pixels_buffer );
    params->pixels = (uchar4*)pixels_buffer->getDevicePointer(optix_device_ordinal); 

    posi_buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, params->width, params->height);
    context["posi_buffer"]->set( posi_buffer );
    params->isect = (float4*)posi_buffer->getDevicePointer(optix_device_ordinal); 
}

void Six::updateContext()
{
    LOG(info); 
    context[ "tmin"]->setFloat( params->tmin );  
    context[ "eye"]->setFloat( params->eye.x, params->eye.y, params->eye.z  );  
    context[ "U"  ]->setFloat( params->U.x, params->U.y, params->U.z  );  
    context[ "V"  ]->setFloat( params->V.x, params->V.y, params->V.z  );  
    context[ "W"  ]->setFloat( params->W.x, params->W.y, params->W.z  );  
    context[ "radiance_ray_type"   ]->setUint( 0u );  
    context[ "cameratype"   ]->setUint( params->cameratype );  
}

template<typename T>
void Six::createContextBuffer( T* d_ptr, unsigned num_item, const char* name )
{
    LOG(info) << name << " " << d_ptr << ( d_ptr == nullptr ? " EMPTY " : "" ); 
    optix::Buffer buffer = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_USER, num_item );
    buffer->setElementSize( sizeof(T) ); 
    if(d_ptr)
    {
        buffer->setDevicePointer(optix_device_ordinal, d_ptr ); 
    }
    context[name]->set( buffer );
}

template void Six::createContextBuffer( CSGNode*,  unsigned, const char* ) ; 
template void Six::createContextBuffer( qat4*,  unsigned, const char* ) ; 
template void Six::createContextBuffer( float*, unsigned, const char* ) ; 


void Six::setFoundry(const CSGFoundry* foundry_)  // HMM: maybe makes more sense to get given directly the lower level CSGFoundry ?
{
    foundry = foundry_ ; 
    create(); 
}
    
void Six::create()
{
    LOG(info) << "[" ; 
    createContextBuffers(); 
    createGAS(); 
    createIAS(); 
    LOG(info) << "]" ; 
}


/**
Six::createContextBuffers
---------------------------

NB the CSGPrim prim_buffer is not here as that is specific 
to each geometry "solid"

These CSGNode float4 planes and qat4 inverse transforms are 
here because those are globally referenced.

**/
void Six::createContextBuffers()
{
    createContextBuffer<CSGNode>(   foundry->d_node, foundry->getNumNode(), "node_buffer" ); 
    createContextBuffer<qat4>(      foundry->d_itra, foundry->getNumItra(), "itra_buffer" ); 
    createContextBuffer<float4>(    foundry->d_plan, foundry->getNumPlan(), "plan_buffer" ); 
}


/**
Six::createGeometry
----------------------

**/

optix::Geometry Six::createGeometry(unsigned solid_idx)
{
    const CSGSolid* so = foundry->solid.data() + solid_idx ; 
    unsigned primOffset = so->primOffset ;  
    unsigned numPrim = so->numPrim ; 
    CSGPrim* d_pr = foundry->d_prim + primOffset ; 

    LOG(info) 
        << " solid_idx " << std::setw(3) << solid_idx
        << " numPrim " << std::setw(3) << numPrim 
        << " primOffset " << std::setw(3) << primOffset
        << " d_pr " << d_pr
        ;

    optix::Geometry solid = context->createGeometry();
    solid->setPrimitiveCount( numPrim );
    solid->setBoundingBoxProgram( context->createProgramFromPTXFile( geo_ptx_path , "bounds" ) );
    solid->setIntersectionProgram( context->createProgramFromPTXFile( geo_ptx_path , "intersect" ) ) ; 

    optix::Buffer prim_buffer = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_USER, numPrim );
    prim_buffer->setElementSize( sizeof(CSGPrim) ); 
    prim_buffer->setDevicePointer(optix_device_ordinal, d_pr ); 
    solid["prim_buffer"]->set( prim_buffer );
 
    return solid ; 
}

void Six::createGAS()
{
    unsigned num_solid = foundry->getNumSolid();   
    LOG(info) << "num_solid " << num_solid ;  

    for(unsigned i=0 ; i < num_solid ; i++)
    {
        optix::Geometry solid = createGeometry(i); 
        solids.push_back(solid); 
    }
}

void Six::createIAS()
{
    unsigned num_ias = foundry->getNumUniqueIAS() ; 
    for(unsigned i=0 ; i < num_ias ; i++)
    {
        unsigned ias_idx = foundry->ias[i]; 
        optix::Group group = createIAS(ias_idx); 
        groups.push_back(group); 
    }
}

optix::Group Six::createIAS(unsigned ias_idx)
{
    unsigned num_inst = foundry->getNumInst(); 
    unsigned ias_inst = foundry->getNumInstancesIAS(ias_idx); 
    LOG(info) 
        << " ias_idx " << ias_idx
        << " num_inst " << num_inst 
        << " ias_inst " << ias_inst 
        ; 
    assert( ias_inst > 0); 

    const char* accel = "Trbvh" ; 
    optix::Acceleration instance_accel = context->createAcceleration(accel);
    optix::Acceleration group_accel  = context->createAcceleration(accel);

    optix::Group group = context->createGroup();
    group->setChildCount( ias_inst );
    group->setAcceleration( group_accel );  

    unsigned count = 0 ; 
    for(unsigned i=0 ; i < num_inst ; i++)
    {
        const qat4* qc = foundry->getInst(i); 
        unsigned ins_idx,  gas_idx, ias_idx_ ;
        qc->getIdentity( ins_idx,  gas_idx, ias_idx_ ); 
        assert( ins_idx == i ); 

        if( ias_idx_ == ias_idx )
        {
            const float* qcf = qc->cdata(); 
            qat4 q(qcf);        // copy to clear identity before passing to OptiX
            q.clearIdentity(); 

            optix::Transform xform = context->createTransform();
            bool transpose = true ; 
            xform->setMatrix(transpose, q.data(), 0); 
            group->setChild(count, xform);

            optix::GeometryInstance pergi = createGeometryInstance(gas_idx, ins_idx); 
            optix::GeometryGroup perxform = context->createGeometryGroup();
            perxform->addChild(pergi); 
            perxform->setAcceleration(instance_accel) ; 
            xform->setChild(perxform);

            count += 1 ; 
        }
    }
    assert( count == ias_inst ); 
    return group ;
}

optix::GeometryInstance Six::createGeometryInstance(unsigned gas_idx, unsigned ins_idx)
{
    //std::cout << "Six::createGeometryInstance" << " gas_idx " << gas_idx << " ins_idx " << ins_idx << std::endl ;   
    optix::Geometry solid = solids[gas_idx]; 

    optix::GeometryInstance pergi = context->createGeometryInstance() ;
    pergi->setMaterialCount(1);
    pergi->setMaterial(0, material );
    pergi->setGeometry(solid);
    pergi["identity"]->setUint(ins_idx);

    return pergi ; 
}


void Six::setTop(const char* spec)
{
    char c = spec[0]; 
    assert( c == 'i' || c == 'g' );  
    int idx = atoi( spec + 1 );  

    LOG(info) << "spec " << spec ; 
    if( c == 'i' )
    {
        assert( idx < groups.size() ); 
        optix::Group grp = groups[idx]; 
        context["top_object"]->set( grp );
    }
    else if( c == 'g' )
    {
        assert( idx < solids.size() ); 

        optix::GeometryGroup gg = context->createGeometryGroup();
        gg->setChildCount(1);
     
        unsigned identity = 1u + idx ;  
        optix::GeometryInstance pergi = createGeometryInstance(idx, identity); 
        gg->setChild( 0, pergi );
        gg->setAcceleration( context->createAcceleration("Trbvh") );

        context["top_object"]->set( gg );
    }
}




void Six::launch()
{
    LOG(info) ; 
    context->launch( entry_point_index , params->width, params->height  );  
}


/*
void Six::save(const char* outdir) 
{
    int channels = 4 ; 
    int quality = 50 ; 

    const unsigned char* data = (const unsigned char*)pixels_buffer->map();  
    SIMG img(int(params->width), int(params->height), channels,  data ); 
    img.writeJPG(outdir, "pixels.jpg", quality); 
    pixels_buffer->unmap(); 

    NP::Write(outdir, "posi.npy",  (float*)posi_buffer->map(), params->height, params->width, 4 );
    posi_buffer->unmap(); 
}
*/


