cmake_minimum_required(VERSION 3.14)
set (CMAKE_CXX_STANDARD 14)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-Werror -Wall -Wextra -fmax-errors=1 -pthread -g")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -DDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
add_compile_definitions(USE_ACCELERGY BUILD_BASE_DIR="${CMAKE_SOURCE_DIR}/lib/timeloop")

project(medea)

# Define CMAKE_INSTALL_xxx: LIBDIR, INCLUDEDIR
include(GNUInstallDirs)

include_directories(
    ${PROJECT_SOURCE_DIR}/src 
    ${PROJECT_SOURCE_DIR}/lib/timeloop/src
    ${PROJECT_SOURCE_DIR}/lib/timeloop/pat-public/src
)

file(GLOB_RECURSE MEDEA_SRCS CONFIGURE_DEPENDS 
        "${PROJECT_SOURCE_DIR}/src/*.?pp" 
        "${PROJECT_SOURCE_DIR}/include/*.?pp"
    )
set(TIMELOOP_SRCS
    "${PROJECT_SOURCE_DIR}/lib/timeloop/src/loop-analysis/tiling.cpp"
    "${PROJECT_SOURCE_DIR}/lib/timeloop/src/loop-analysis/nest-analysis.cpp"
    "${PROJECT_SOURCE_DIR}/lib/timeloop/pat-public/src/pat/pat.cpp"
    "${PROJECT_SOURCE_DIR}/lib/timeloop/src/mapping/loop.cpp"
    "${PROJECT_SOURCE_DIR}/lib/timeloop/src/mapping/nest.cpp"
    "${PROJECT_SOURCE_DIR}/lib/timeloop/src/model/arithmetic.cpp"
    "${PROJECT_SOURCE_DIR}/lib/timeloop/src/model/buffer.cpp"
    "${PROJECT_SOURCE_DIR}/lib/timeloop/src/model/topology.cpp"
    "${PROJECT_SOURCE_DIR}/lib/timeloop/src/model/network-legacy.cpp"
    "${PROJECT_SOURCE_DIR}/lib/timeloop/src/model/network-reduction-tree.cpp"
    "${PROJECT_SOURCE_DIR}/lib/timeloop/src/model/network-simple-multicast.cpp"
    "${PROJECT_SOURCE_DIR}/lib/timeloop/src/util/numeric.cpp"
    "${PROJECT_SOURCE_DIR}/lib/timeloop/src/util/map2d.cpp"
    "${PROJECT_SOURCE_DIR}/lib/timeloop/src/workload/problem-shape.cpp"
    "${PROJECT_SOURCE_DIR}/lib/timeloop/src/workload/workload.cpp"
    "${PROJECT_SOURCE_DIR}/lib/timeloop/src/workload/operation-space.cpp"
    "${PROJECT_SOURCE_DIR}/lib/timeloop/src/compound-config/compound-config.cpp"
    "${PROJECT_SOURCE_DIR}/lib/timeloop/src/data/cnn/cnn-layers.cpp"
    "${PROJECT_SOURCE_DIR}/lib/timeloop/src/mapping/mapping.cpp"
    "${PROJECT_SOURCE_DIR}/lib/timeloop/src/mapping/parser.cpp"
    "${PROJECT_SOURCE_DIR}/lib/timeloop/src/mapspaces/mapspace-base.cpp"
)
add_executable(medea ${MEDEA_SRCS} ${TIMELOOP_SRCS})

# LIBCONFIG++
find_library(CONFIG++_LIBRARY NAMES config++ PATH ${LIBDIR})
include_directories(${INCLUDEDIR})

# YAML
find_package(yaml-cpp REQUIRED)
include_directories(${YAML_CPP_INCLUDE_DIRS})

# BOOST
set(Boost_USE_STATIC_LIBS ON) 
set(Boost_USE_MULTITHREADED ON)  
set(Boost_USE_STATIC_RUNTIME OFF) 
find_package(Boost REQUIRED COMPONENTS iostreams serialization program_options)
message(STATUS "Boost version: ${Boost_VERSION}") 
include_directories(${Boost_INCLUDE_DIRS}) 

# Linking
target_link_libraries(medea ${Boost_LIBRARIES} ${YAML_CPP_LIBRARIES} ${CONFIG++_LIBRARY})
target_link_options(medea PRIVATE -static-libgcc -static-libstdc++)

install(TARGETS medea RUNTIME DESTINATION bin)