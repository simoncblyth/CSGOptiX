#include <iostream>

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "Params.h"
#include "InstanceId.h"
#include "Geo.h"

#include "sutil_vec_math.h"
#include "CSGFoundry.h"
#include "CSGSolid.h"
#include "CSGPrim.h"
#include "OpticksCSG.h"
#include "CSGNode.h"


#include "Grid.h"
#include "Six.h"

#include "SIMG.hh" 
#include "NP.hh"
    
Six::Six(const char* ptx_path_, const Params* params_)
    :
    context(optix::Context::create()),
    material(context->createMaterial()),
    params(params_),
    ptx_path(strdup(ptx_path_)),
    entry_point_index(0u),
    optix_device_ordinal(0u),
    geo(nullptr),
    foundry(nullptr)
{
    initContext();
    initPipeline(); 
}

void Six::initContext()
{
    context->setRayTypeCount(1);
    context->setPrintEnabled(true);
    context->setPrintBufferSize(4096);
    context->setPrintLaunchIndex(0); 
    context->setEntryPointCount(1);

    context[ "tmin"]->setFloat( params->tmin );  
    context[ "eye"]->setFloat( params->eye.x, params->eye.y, params->eye.z  );  
    context[ "U"  ]->setFloat( params->U.x, params->U.y, params->U.z  );  
    context[ "V"  ]->setFloat( params->V.x, params->V.y, params->V.z  );  
    context[ "W"  ]->setFloat( params->W.x, params->W.y, params->W.z  );  
    context[ "radiance_ray_type"   ]->setUint( 0u );  
    context[ "cameratype"   ]->setUint( params->cameratype );  

    pixels_buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_UNSIGNED_BYTE4, params->width, params->height);
    context["pixels_buffer"]->set( pixels_buffer );
    posi_buffer = context->createBuffer( RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, params->width, params->height);
    context["posi_buffer"]->set( posi_buffer );
}

void Six::initPipeline()
{
    context->setRayGenerationProgram( entry_point_index, context->createProgramFromPTXFile( ptx_path , "raygen" ));
    context->setMissProgram(   entry_point_index, context->createProgramFromPTXFile( ptx_path , "miss" ));

    material->setClosestHitProgram( entry_point_index, context->createProgramFromPTXFile( ptx_path, "closest_hit" ));
}

template<typename T>
void Six::createContextBuffer( T* d_ptr, unsigned num_item, const char* name )
{
    std::cout << "[ Six::createContextBuffer " << name << " " << d_ptr << std::endl ; 
    optix::Buffer buffer = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_USER, num_item );
    buffer->setElementSize( sizeof(T) ); 
    if(d_ptr)
    {
        buffer->setDevicePointer(optix_device_ordinal, d_ptr ); 
    }
    context[name]->set( buffer );
    std::cout << "] Six::createContextBuffer " << name << std::endl ; 
}

template void Six::createContextBuffer( CSGNode*,  unsigned, const char* ) ; 
template void Six::createContextBuffer( qat4*,  unsigned, const char* ) ; 
template void Six::createContextBuffer( float*, unsigned, const char* ) ; 


/**

Six::setGeo
-------------

Needs generalization, get away from the specific example of Geo/Grids

Top object mechanics should not be here.
**/


void Six::setGeo(const Geo* geo_)  // HMM: maybe makes more sense to get given directly the lower level CSGFoundry ?
{
    geo = geo_ ; 
    foundry = geo->foundry ; 
    convert(); 
}
    
void Six::convert()
{
    unsigned num_solid = foundry->getNumSolid(); 
    std::cout << "Six::convert num_solid " << num_solid << std::endl ;  

    createContextBuffer<CSGNode>(   foundry->d_node, foundry->getNumNode(), "node_buffer" ); 
    createContextBuffer<qat4>(   foundry->d_itra, foundry->getNumItra(), "itra_buffer" ); 
    createContextBuffer<float4>( foundry->d_plan, foundry->getNumPlan(), "plan_buffer" ); 
    // these "global" context buffers have no offsets

    convertSolids(); 
    convertGroups(); 

    const char* spec = geo->top ;  
    char c = spec[0]; 
    assert( c == 'i' || c == 'g' );  
    int idx = atoi( spec + 1 );  

    std::cout << "Six::setGeo spec " << spec << std::endl ; 
    if( c == 'i' )
    {
        assert( idx < assemblies.size() ); 
        optix::Group grp = assemblies[idx]; 
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

void Six::convertSolids()
{
    unsigned num_solid = foundry->getNumSolid();   // just pass thru to foundry  
    std::cout << "Six::createShapes num_solid " << num_solid << std::endl ;  

    for(unsigned i=0 ; i < num_solid ; i++)
    {
        optix::Geometry solid = createGeometry(i); 
        solids.push_back(solid); 
    }
}

void Six::convertGroups()
{
    unsigned num_ias = foundry->ias.size() ; 
    for(unsigned i=0 ; i < num_ias ; i++)
    {
        unsigned ias_idx = foundry->ias[i]; 
        optix::Group assembly = createAssembly(ias_idx); 
        assemblies.push_back(assembly); 
    }
}

optix::Group Six::createAssembly(unsigned ias)
{
    unsigned num_inst = foundry->getNumInst(); 
    unsigned ias_inst = foundry->getNumInstancesIAS(ias); 
    std::cout 
        << "Six::createAssembly"
        << " num_inst " << num_inst 
        << " ias " << ias
        << " ias_inst " << ias_inst 
        << std::endl
        ; 
    assert( ias_inst > 0); 

    const char* accel = "Trbvh" ; 
    optix::Acceleration instance_accel = context->createAcceleration(accel);
    optix::Acceleration assembly_accel  = context->createAcceleration(accel);

    optix::Group assembly = context->createGroup();
    assembly->setChildCount( ias_inst );
    assembly->setAcceleration( assembly_accel );  

    unsigned count = 0 ; 
    for(unsigned i=0 ; i < num_inst ; i++)
    {
        const qat4* qc = foundry->getInst(i); 
        unsigned ins_idx,  gas_idx, ias_idx ;
        qc->getIdentity( ins_idx,  gas_idx, ias_idx ); 
        assert( ins_idx == i ); 

        if( ias_idx == ias )
        {
            const float* qcf = qc->cdata(); 
            qat4 q(qcf);        // copy to clear identity before passing to OptiX
            q.clearIdentity(); 

            optix::Transform xform = context->createTransform();
            bool transpose = true ; 
            xform->setMatrix(transpose, q.data(), 0); 
            assembly->setChild(count, xform);

            optix::GeometryInstance pergi = createGeometryInstance(gas_idx, ins_idx); 
            optix::GeometryGroup perxform = context->createGeometryGroup();
            perxform->addChild(pergi); 
            perxform->setAcceleration(instance_accel) ; 
            xform->setChild(perxform);

            count += 1 ; 
        }
    }
    assert( count == ias_inst ); 

    return assembly ;
}

optix::GeometryInstance Six::createGeometryInstance(unsigned gas_idx, unsigned ins_idx)
{
    std::cout 
        << "Six::createGeometryInstance"
        << " gas_idx " << gas_idx
        << " ins_idx " << ins_idx
        << std::endl 
        ;   

    optix::Geometry solid = solids[gas_idx]; 

    optix::GeometryInstance pergi = context->createGeometryInstance() ;
    pergi->setMaterialCount(1);
    pergi->setMaterial(0, material );
    pergi->setGeometry(solid);
    pergi["identity"]->setUint(ins_idx);

    return pergi ; 
}

optix::Geometry Six::createGeometry(unsigned solid_idx)
{
    const CSGSolid* so = foundry->solid.data() + solid_idx ; 
    unsigned primOffset = so->primOffset ;  
    unsigned numPrim = so->numPrim ; 
    CSGPrim* d_pr = foundry->d_prim + primOffset ; 

    std::cout 
        << "Six::createSolidGeometry"
        << " solid_idx " << solid_idx
        << " primOffset " << primOffset
        << " numPrim " << numPrim 
        << std::endl 
        ;

    optix::Geometry solid = context->createGeometry();
    solid->setPrimitiveCount( numPrim );
    solid->setBoundingBoxProgram( context->createProgramFromPTXFile( ptx_path , "bounds" ) );
    solid->setIntersectionProgram( context->createProgramFromPTXFile( ptx_path , "intersect" ) ) ; 

    optix::Buffer prim_buffer = context->createBuffer( RT_BUFFER_INPUT, RT_FORMAT_USER, numPrim );
    prim_buffer->setElementSize( sizeof(CSGPrim) ); 
    prim_buffer->setDevicePointer(optix_device_ordinal, d_pr ); 
    solid["prim_buffer"]->set( prim_buffer );
 
    return solid ; 
}

void Six::launch()
{
    context->launch( entry_point_index , params->width, params->height  );  
}

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

