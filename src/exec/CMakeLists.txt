cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TARGET exec)
add_library(${TARGET} SHARED query_synthesizer.cc index.cc filter_allocator.cc node.cc buffer.cc eapca.cc split.cc answer.cc)

set_target_properties(${TARGET} PROPERTIES LINKER_LANGUAGE CXX)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_compile_options(${TARGET} PUBLIC -march=native -msse3 -msse4.2 -mavx -mavx2)

target_link_libraries(${TARGET}
        PRIVATE navigator filter model util common ${Boost_LIBRARIES} GSL::gsl GSL::gslcblas ${TORCH_LIBRARIES} pthread
        )
