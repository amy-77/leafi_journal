cmake_minimum_required(VERSION 3.16)

set(TARGET util)
add_library(${TARGET}
        SHARED stat.cc distance.h interval.h vec.h sort.cc comp.h str.h)

set_target_properties(${TARGET} PROPERTIES LINKER_LANGUAGE CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

target_compile_features(${TARGET} PRIVATE cxx_std_17)

# -mavx -mavx2 -mavx512 -msse4.2 are supported by Intel(R) Xeon(R) Gold 6134 CPU @ 3.20GHz in andromache
#target_compile_options(utils PUBLIC -march=native -msse4.2 -mavx)
target_compile_options(${TARGET} PUBLIC -march=native -msse3 -msse4.2 -mavx -mavx2)

target_link_libraries(${TARGET}
        PRIVATE common
        )
