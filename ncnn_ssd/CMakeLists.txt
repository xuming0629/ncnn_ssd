cmake_minimum_required(VERSION 2.8.12)
include(${PROJECT_SOURCE_DIR}/cmake/libncnn.cmake)
# set(NCNN_DIR "E:/workspaces/NCNN/project/ncnn_ssd/ThirdParty/ncnn_cpu")
include_directories(# ${NCNN_DIR}/include/ncnn
                    ${PROJECT_SOURCE_DIR}/ncnn_ssd
)
# ncnn
# add_library(ncnn STATIC IMPORTED)
# set_target_properties(
#     ncnn
#     PROPERTIES IMPORTED_LOCATION
#     ${NCNN_DIR}/lib/ncnn.lib
# )
# set_target_properties(ncnn_ssd_static PROPERTIES OUTPUT_NAME ncnn_ssd)

file(GLOB SRC_FILE ${PROJECT_SOURCE_DIR}/ncnn_ssd/*.cpp)
message(STATUS " this is src dir " ${SRC_FILE})
set(SSD_SRC ${SRC_FILE} )
# add_executable(ncnn_ssd_demo ${SSD_SRC})
add_library(ncnn_ssd STATIC ${SSD_SRC})
target_link_libraries(ncnn_ssd
                      ncnn
)
INSTALL(TARGETS ncnn_ssd 
        ARCHIVE DESTINATION lib 
)
INSTALL(FILES SSDDetector.h common.h BaseDetector.h DESTINATION include/)