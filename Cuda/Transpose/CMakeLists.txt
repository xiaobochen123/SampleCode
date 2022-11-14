cmake_minimum_required(VERSION 3.18.2)

project(Transpose LANGUAGES CXX CUDA)

add_executable(${PROJECT_NAME}
    main.cu
)

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_ARCHITECTURES ${CUDA_ARCHS})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../Common/
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(${PROJECT_NAME}
    ${CUDART_LIBRARY}
    ${CUBLAS_LIBRARY}
    pybind11::embed
)

target_compile_options(${PROJECT_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)