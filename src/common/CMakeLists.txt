cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TARGET common)
add_library(${TARGET} SHARED config.cc global.cc)

set_target_properties(${TARGET} PROPERTIES LINKER_LANGUAGE CXX)
target_compile_features(${TARGET} PRIVATE cxx_std_14)
#set_target_properties(${TARGET} PROPERTIES COMPILE_DEFINITIONS "BOOST_LOG_DYN_LINK")

target_link_libraries(${TARGET}
        PUBLIC ${TORCH_LIBRARIES} ${Boost_LIBRARIES} spdlog::spdlog
        )
