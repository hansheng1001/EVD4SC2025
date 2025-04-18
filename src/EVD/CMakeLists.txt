
file(GLOB CODE_SOURCES "*.cu")


foreach(SRC ${CODE_SOURCES})
    get_filename_component(NAME ${SRC} NAME_WE)
    message(${NAME})
    add_executable(${NAME} "${NAME}.cu")
    set_property(TARGET ${NAME}
            PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    
    target_include_directories(${NAME} PRIVATE ${CMAKE_SOURCE_DIR}/util/inc)
    target_include_directories(${NAME} PRIVATE ${CMAKE_SOURCE_DIR}/api/inc)
    target_include_directories(${NAME} PRIVATE ${CMAKE_SOURCE_DIR}/inc)

    target_link_libraries(${NAME} my_api_shared_lib)
    target_link_libraries(${NAME} curand cusolver cublas)


    if(${NAME} STREQUAL "myEVD")
        
        target_link_libraries(${NAME} mkl_gf_ilp64 mkl_gnu_thread mkl_core gomp "-fopenmp")
        
    endif()
    

endforeach()