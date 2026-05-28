cmake_minimum_required(VERSION 3.20)
project(ml_labs)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/math
)

add_executable(laba1
        main.cpp
        math/vector/vector.cpp
        math/matrix/matrix.cpp
        math/random/random.cpp
        generator/generator.cpp
        classifier/classifier.cpp
        geneticalg/geneticalg.cpp
        neural/neuralnetwork.cpp
        metrics/f1score.h
        metrics/f1score.cpp
        data/dataloader.h
        data/dataloader.cpp
        test_load.cpp
)
