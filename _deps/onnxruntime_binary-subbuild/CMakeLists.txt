# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.27.7)

# Reject any attempt to use a toolchain file. We must not use one because
# we could be downloading it here. If the CMAKE_TOOLCHAIN_FILE environment
# variable is set, the cache variable will have been initialized from it.
unset(CMAKE_TOOLCHAIN_FILE CACHE)
unset(ENV{CMAKE_TOOLCHAIN_FILE})

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(onnxruntime_binary-populate NONE)



include(ExternalProject)
ExternalProject_Add(onnxruntime_binary-populate
                     "UPDATE_DISCONNECTED" "False" "EXTERNALPROJECT_INTERNAL_ARGUMENT_SEPARATOR" "URL" "https://github.com/microsoft/onnxruntime/releases/download/v1.12.1/onnxruntime-linux-x64-1.12.1.tgz" "TLS_VERIFY" "TRUE" "DOWNLOAD_DIR" "/home/kevin/Projects/cvops/cvops-python/cvops-inference/external/onnx/build/external/onnxruntime/download" DOWNLOAD_NO_EXTRACT YES
                    SOURCE_DIR          "/home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-src"
                    BINARY_DIR          "/home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
                    USES_TERMINAL_PATCH     YES
)


ExternalProject_Get_Property(onnxruntime_binary-populate DOWNLOADED_FILE)
get_filename_component(dlFileName "${DOWNLOADED_FILE}" NAME)

ExternalProject_Add_Step(onnxruntime_binary-populate copyfile
  COMMAND    "/home/kevin/Projects/cvops/cvops-python/venv/lib/python3.10/site-packages/cmake/data/bin/cmake" -E copy_if_different
             "<DOWNLOADED_FILE>" "/home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-src"
  DEPENDEES  patch
  DEPENDERS  configure
  BYPRODUCTS "/home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-src/${dlFileName}"
  COMMENT    "Copying file to SOURCE_DIR"
)

