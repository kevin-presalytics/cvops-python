# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.27

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /home/kevin/Projects/cvops/cvops-python/venv/lib/python3.10/site-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /home/kevin/Projects/cvops/cvops-python/venv/lib/python3.10/site-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-subbuild

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-subbuild

# Utility rule file for onnxruntime_binary-populate.

# Include any custom commands dependencies for this target.
include CMakeFiles/onnxruntime_binary-populate.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/onnxruntime_binary-populate.dir/progress.make

CMakeFiles/onnxruntime_binary-populate: CMakeFiles/onnxruntime_binary-populate-complete

CMakeFiles/onnxruntime_binary-populate-complete: onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-install
CMakeFiles/onnxruntime_binary-populate-complete: onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-mkdir
CMakeFiles/onnxruntime_binary-populate-complete: onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-download
CMakeFiles/onnxruntime_binary-populate-complete: onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-update
CMakeFiles/onnxruntime_binary-populate-complete: onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-patch
CMakeFiles/onnxruntime_binary-populate-complete: onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-configure
CMakeFiles/onnxruntime_binary-populate-complete: onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-build
CMakeFiles/onnxruntime_binary-populate-complete: onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-install
CMakeFiles/onnxruntime_binary-populate-complete: onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-test
CMakeFiles/onnxruntime_binary-populate-complete: onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-copyfile
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Completed 'onnxruntime_binary-populate'"
	/home/kevin/Projects/cvops/cvops-python/venv/lib/python3.10/site-packages/cmake/data/bin/cmake -E make_directory /home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-subbuild/CMakeFiles
	/home/kevin/Projects/cvops/cvops-python/venv/lib/python3.10/site-packages/cmake/data/bin/cmake -E touch /home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-subbuild/CMakeFiles/onnxruntime_binary-populate-complete
	/home/kevin/Projects/cvops/cvops-python/venv/lib/python3.10/site-packages/cmake/data/bin/cmake -E touch /home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-subbuild/onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-done

onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-build: onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-configure
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "No build step for 'onnxruntime_binary-populate'"
	cd /home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-build && /home/kevin/Projects/cvops/cvops-python/venv/lib/python3.10/site-packages/cmake/data/bin/cmake -E echo_append
	cd /home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-build && /home/kevin/Projects/cvops/cvops-python/venv/lib/python3.10/site-packages/cmake/data/bin/cmake -E touch /home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-subbuild/onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-build

onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-configure: onnxruntime_binary-populate-prefix/tmp/onnxruntime_binary-populate-cfgcmd.txt
onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-configure: onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-patch
onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-configure: onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-copyfile
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "No configure step for 'onnxruntime_binary-populate'"
	cd /home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-build && /home/kevin/Projects/cvops/cvops-python/venv/lib/python3.10/site-packages/cmake/data/bin/cmake -E echo_append
	cd /home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-build && /home/kevin/Projects/cvops/cvops-python/venv/lib/python3.10/site-packages/cmake/data/bin/cmake -E touch /home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-subbuild/onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-configure

onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-copyfile: onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-patch
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Copying file to SOURCE_DIR"
	/home/kevin/Projects/cvops/cvops-python/venv/lib/python3.10/site-packages/cmake/data/bin/cmake -E copy_if_different /home/kevin/Projects/cvops/cvops-python/cvops-inference/external/onnx/build/external/onnxruntime/download/onnxruntime-linux-x64-1.12.1.tgz /home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-src
	/home/kevin/Projects/cvops/cvops-python/venv/lib/python3.10/site-packages/cmake/data/bin/cmake -E touch /home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-subbuild/onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-copyfile

onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-download: onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/download-onnxruntime_binary-populate.cmake
onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-download: onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-urlinfo.txt
onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-download: onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-mkdir
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Performing download step (download and verify) for 'onnxruntime_binary-populate'"
	cd /home/kevin/Projects/cvops/cvops-python/_deps && /home/kevin/Projects/cvops/cvops-python/venv/lib/python3.10/site-packages/cmake/data/bin/cmake -P /home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-subbuild/onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/download-onnxruntime_binary-populate.cmake
	cd /home/kevin/Projects/cvops/cvops-python/_deps && /home/kevin/Projects/cvops/cvops-python/venv/lib/python3.10/site-packages/cmake/data/bin/cmake -P /home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-subbuild/onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/verify-onnxruntime_binary-populate.cmake
	cd /home/kevin/Projects/cvops/cvops-python/_deps && /home/kevin/Projects/cvops/cvops-python/venv/lib/python3.10/site-packages/cmake/data/bin/cmake -E touch /home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-subbuild/onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-download

onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-install: onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-build
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "No install step for 'onnxruntime_binary-populate'"
	cd /home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-build && /home/kevin/Projects/cvops/cvops-python/venv/lib/python3.10/site-packages/cmake/data/bin/cmake -E echo_append
	cd /home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-build && /home/kevin/Projects/cvops/cvops-python/venv/lib/python3.10/site-packages/cmake/data/bin/cmake -E touch /home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-subbuild/onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-install

onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-mkdir:
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_7) "Creating directories for 'onnxruntime_binary-populate'"
	/home/kevin/Projects/cvops/cvops-python/venv/lib/python3.10/site-packages/cmake/data/bin/cmake -Dcfgdir= -P /home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-subbuild/onnxruntime_binary-populate-prefix/tmp/onnxruntime_binary-populate-mkdirs.cmake
	/home/kevin/Projects/cvops/cvops-python/venv/lib/python3.10/site-packages/cmake/data/bin/cmake -E touch /home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-subbuild/onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-mkdir

onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-patch: onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-patch-info.txt
onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-patch: onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-update
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_8) "No patch step for 'onnxruntime_binary-populate'"
	/home/kevin/Projects/cvops/cvops-python/venv/lib/python3.10/site-packages/cmake/data/bin/cmake -E echo_append
	/home/kevin/Projects/cvops/cvops-python/venv/lib/python3.10/site-packages/cmake/data/bin/cmake -E touch /home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-subbuild/onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-patch

onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-test: onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-install
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_9) "No test step for 'onnxruntime_binary-populate'"
	cd /home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-build && /home/kevin/Projects/cvops/cvops-python/venv/lib/python3.10/site-packages/cmake/data/bin/cmake -E echo_append
	cd /home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-build && /home/kevin/Projects/cvops/cvops-python/venv/lib/python3.10/site-packages/cmake/data/bin/cmake -E touch /home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-subbuild/onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-test

onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-update: onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-update-info.txt
onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-update: onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-download
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --blue --bold --progress-dir=/home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-subbuild/CMakeFiles --progress-num=$(CMAKE_PROGRESS_10) "No update step for 'onnxruntime_binary-populate'"
	/home/kevin/Projects/cvops/cvops-python/venv/lib/python3.10/site-packages/cmake/data/bin/cmake -E echo_append
	/home/kevin/Projects/cvops/cvops-python/venv/lib/python3.10/site-packages/cmake/data/bin/cmake -E touch /home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-subbuild/onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-update

onnxruntime_binary-populate: CMakeFiles/onnxruntime_binary-populate
onnxruntime_binary-populate: CMakeFiles/onnxruntime_binary-populate-complete
onnxruntime_binary-populate: onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-build
onnxruntime_binary-populate: onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-configure
onnxruntime_binary-populate: onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-copyfile
onnxruntime_binary-populate: onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-download
onnxruntime_binary-populate: onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-install
onnxruntime_binary-populate: onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-mkdir
onnxruntime_binary-populate: onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-patch
onnxruntime_binary-populate: onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-test
onnxruntime_binary-populate: onnxruntime_binary-populate-prefix/src/onnxruntime_binary-populate-stamp/onnxruntime_binary-populate-update
onnxruntime_binary-populate: CMakeFiles/onnxruntime_binary-populate.dir/build.make
.PHONY : onnxruntime_binary-populate

# Rule to build all files generated by this target.
CMakeFiles/onnxruntime_binary-populate.dir/build: onnxruntime_binary-populate
.PHONY : CMakeFiles/onnxruntime_binary-populate.dir/build

CMakeFiles/onnxruntime_binary-populate.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/onnxruntime_binary-populate.dir/cmake_clean.cmake
.PHONY : CMakeFiles/onnxruntime_binary-populate.dir/clean

CMakeFiles/onnxruntime_binary-populate.dir/depend:
	cd /home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-subbuild && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-subbuild /home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-subbuild /home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-subbuild /home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-subbuild /home/kevin/Projects/cvops/cvops-python/_deps/onnxruntime_binary-subbuild/CMakeFiles/onnxruntime_binary-populate.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/onnxruntime_binary-populate.dir/depend
