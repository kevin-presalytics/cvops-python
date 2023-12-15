cmake_minimum_required(VERSION 3.15)

project(cvops-python)


# Compiler settings
SET (CMAKE_C_COMPILER             "/usr/bin/clang")
SET (CMAKE_C_FLAGS                "-Wall -std=c99")
SET (CMAKE_C_FLAGS_DEBUG          "-g")
SET (CMAKE_C_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
SET (CMAKE_C_FLAGS_RELEASE        "-O4 -DNDEBUG")
SET (CMAKE_C_FLAGS_RELWITHDEBINFO "-O2 -g")

SET (CMAKE_CXX_COMPILER             "/usr/bin/clang++")
SET (CMAKE_CXX_FLAGS                "-Wall")
SET (CMAKE_CXX_FLAGS_DEBUG          "-g")
SET (CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELEASE        "-O4 -DNDEBUG")
SET (CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g")

if (UNIX)
    set(PLATFORM "linux")
ELSEIF (WIN32)
    set(PLATFORM "win32")
ELSEIF (APPLE)
    set(PLATFORM "macos")
ENDIF ()

set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)


add_subdirectory(cvops-inference)

SET(PYTHON_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/cvops/lib/${PLATFORM}/${CMAKE_HOST_SYSTEM_PROCESSOR})

install(DIRECTORY cvops-inference/build/lib/ DESTINATION ${PYTHON_LIB_DIR})