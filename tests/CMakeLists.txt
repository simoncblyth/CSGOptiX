cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
set(name CSGOptiXTest)
project(${name} VERSION 0.1.0)

set(TEST_SOURCES
    CSGOptiXTest.cc
)

set(DEMO_SOURCES 
    DemoGeo.cc
    DemoGrid.cc
    DemoUtil.cc
)

foreach(SRC ${TEST_SOURCES})
    get_filename_component(TGT ${SRC} NAME_WE)
    add_executable(${TGT} ${SRC} ${DEMO_SOURCES})

    add_test(${name}.${TGT} ${TGT})

    target_link_libraries(${TGT} CSGOptiX)
    install(TARGETS ${TGT} DESTINATION lib)
endforeach()


