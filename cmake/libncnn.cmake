cmake_minimum_required(VERSION 2.8.12)
set(NCNN_OPENMP ON)
set(NCNN_VULKAN OFF)
if(NCNN_OPENMP)
    find_package(OpenMP)
endif()
if(NCNN_VULKAN)
    find_package(Vulkan REQUIRED)
endif()

set(NCNN_DIR  ${PROJECT_SOURCE_DIR}/ThirdParty/ncnn)


add_library(ncnn STATIC IMPORTED)
if(WIN32)
    include_directories(${NCNN_DIR}/win/include/ncnn)
    set_target_properties(
        ncnn
        PROPERTIES IMPORTED_LOCATION
        ${NCNN_DIR}/win/lib/ncnn.lib
        INTERFACE_LINK_LIBRARIES "Vulkan::Vulkan;OpenMP::OpenMP_CXX"
    )
elseif(ANDROID)
    include_directories(${NCNN_DIR}/android/${CMAKE_ANDROID_ARCH_ABI}/include/ncnn)
    message(STATUS " this is android ncnn include dir " ${NCNN_DIR}/android/${CMAKE_ANDROID_ARCH_ABI}/include/ncnn)
    set_target_properties(
        ncnn
        PROPERTIES IMPORTED_LOCATION
        ${NCNN_DIR}/android/${CMAKE_ANDROID_ARCH_ABI}/lib/libncnn.a
        INTERFACE_LINK_LIBRARIES "Vulkan::Vulkan;OpenMP::OpenMP_CXX"
)
endif()

