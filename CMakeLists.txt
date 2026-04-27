cmake_minimum_required(VERSION 3.20)
project(laba1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
        ${CMAKE_SOURCE_DIR}/math
        ${CMAKE_SOURCE_DIR}
)

add_executable(laba1
        main.cpp
        math/vector/vector.cpp
        math/matrix/matrix.cpp
        math/random/random.cpp
        generator/generator.cpp
        classifier/classifier.h
        classifier/classifier.cpp
        geneticalg/geneticalg.h
        geneticalg/geneticalg.cpp
)

add_executable(lab2 
    main_lab2.cpp
    generator/generator.cpp
    math/vector/vector.cpp
    math/matrix/matrix.cpp
    math/random/random.cpp
    neural/neuralnetwork.cpp
)

target_include_directories(lab2 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})