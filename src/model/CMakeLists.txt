cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TARGET model)
add_library(${TARGET} SHARED models.cc scheduler.cc dataset.h)

set_target_properties(${TARGET} PROPERTIES LINKER_LANGUAGE CXX)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
#target_compile_options(${TARGET} PUBLIC -march=native -msse3 -msse4.2 -mavx -mavx2)

target_link_libraries(${TARGET}
        PRIVATE common ${TORCH_LIBRARIES}
        )
