cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(TARGET filter)
add_library(${TARGET} SHARED filter.cc conformal.cc)

set_target_properties(${TARGET} PROPERTIES LINKER_LANGUAGE CXX)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
#target_compile_options(${TARGET} PUBLIC -march=native -msse3 -msse4.2 -mavx -mavx2)

# 添加Eigen头文件路径
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/third_party/eigen)

# 添加dlib源码路径
include_directories(${CMAKE_SOURCE_DIR}/third_party)

# 启用RVM宏（代替GAM）
add_definitions(-DUSE_DLIB_GAM)  # 保持宏名称不变，但功能已改为使用RVM

# 添加dlib所需的编译设置
add_definitions(-DDLIB_NO_GUI_SUPPORT)
add_definitions(-DDLIB_USE_BLAS)
add_definitions(-DDLIB_USE_LAPACK)
add_definitions(-DDLIB_NO_GUI_SUPPORT)
add_definitions(-DDLIB_USE_CUDA=0)  # 确保不使用CUDA，简化编译

target_link_libraries(${TARGET}
        PRIVATE model util common GSL::gsl GSL::gslcblas ${TORCH_LIBRARIES}
        )
