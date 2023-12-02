# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-src"
  "/home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-build"
  "/home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-subbuild/onnxruntime_binary-populate-prefix"
  "/home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-subbuild/onnxruntime_binary-populate-prefix/tmp"
  "/home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-subbuild/onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp"
  "/home/kevin/Projects/cvops/cvops-python/cvops-inference/external/onnx/build/external/onnxruntime/download"
  "/home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-subbuild/onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-subbuild/onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-subbuild/onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()
