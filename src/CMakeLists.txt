

set(UTILS_SOURCES_DIR ${CMAKE_SOURCE_DIR}/util)
file(GLOB_RECURSE UTILS_SOURCES "${UTILS_SOURCES_DIR}/*.cpp" "${UTILS_SOURCES_DIR}/*.cu")
add_library(my_utils_shared_lib SHARED  ${UTILS_SOURCES})

target_include_directories(my_utils_shared_lib PRIVATE ${CMAKE_SOURCE_DIR}/util/inc)

target_link_libraries(my_utils_shared_lib cublas curand)


set(API_SOURCES_DIR ${CMAKE_SOURCE_DIR}/api)
file(GLOB_RECURSE API_SOURCES "${API_SOURCES_DIR}/*.cpp" "${API_SOURCES_DIR}/*.cu")
message("${API_SOURCES}")
add_library(my_api_shared_lib SHARED ${API_SOURCES})

target_include_directories(my_api_shared_lib PRIVATE ${CMAKE_SOURCE_DIR}/util/inc)
target_include_directories(my_api_shared_lib PRIVATE ${CMAKE_SOURCE_DIR}/api/inc)
target_include_directories(my_api_shared_lib PRIVATE ${CMAKE_SOURCE_DIR}/inc)
target_link_libraries(my_api_shared_lib PUBLIC my_utils_shared_lib)


add_executable(my_SB2TR_ZY_ZY_V3 my_SB2TR_ZY_ZY_V3.cu)

target_link_libraries(my_SB2TR_ZY_ZY_V3 my_api_shared_lib)

target_include_directories(my_SB2TR_ZY_ZY_V3 PRIVATE ${CMAKE_SOURCE_DIR}/util/inc)
target_include_directories(my_SB2TR_ZY_ZY_V3 PRIVATE ${CMAKE_SOURCE_DIR}/api/inc)
target_include_directories(my_SB2TR_ZY_ZY_V3 PRIVATE ${CMAKE_SOURCE_DIR}/inc)

target_link_libraries(my_SB2TR_ZY_ZY_V3 curand cusolver cublas)
