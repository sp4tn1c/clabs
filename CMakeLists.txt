cmake_minimum_required(VERSION 3.20)
project(ml_labs)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/math
)

# ==================== C++ программа (laba1) ====================
add_executable(laba1
        main.cpp
        math/vector/vector.cpp
        math/matrix/matrix.cpp
        math/random/random.cpp
        generator/generator.cpp
        classifier/classifier.cpp
        geneticalg/geneticalg.cpp
        neural/neuralnetwork.cpp
        neural/neural_trainer.cpp
        metrics/f1score.cpp
        data/dataloader.cpp
)

# ==================== Python модуль (ml_model_cpp) ====================
find_package(pybind11 CONFIG QUIET)

if(pybind11_FOUND)
    pybind11_add_module(ml_model_cpp
            pybind_wrapper.cpp
            math/vector/vector.cpp
            math/matrix/matrix.cpp
            math/random/random.cpp
            generator/generator.cpp
            neural/neuralnetwork.cpp
            neural/neural_trainer.cpp
            data/dataloader.cpp
            metrics/f1score.cpp
    )

    target_include_directories(ml_model_cpp PRIVATE
            ${CMAKE_SOURCE_DIR}
            ${CMAKE_SOURCE_DIR}/math
    )
else()
    message(WARNING "pybind11 not found - Python module will not be built")
endif()