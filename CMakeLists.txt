cmake_minimum_required(VERSION 3.10)
project(WasmFESolver)

set(CMAKE_CXX_STANDARD 17)

# Emscripten specific settings
if (EMSCRIPTEN)
    set(CMAKE_EXECUTABLE_SUFFIX ".js")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s WASM=1 -s EXPORTED_FUNCTIONS='[\"_solve\"]' -s EXPORTED_RUNTIME_METHODS='[\"ccall\", \"cwrap\"]'")
endif()

add_executable(fesolver 
    src/main.cpp 
    src/solver.cpp
    src/element.cpp
) 