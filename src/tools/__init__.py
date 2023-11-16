# This file contains helper functions for the build process
import subprocess
import pathlib
import shutil
import unittest
import sys


ROOT_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()
C_SOURCE_DIR = ROOT_DIR.joinpath("cvops-inference")
C_BUILD_DIR = C_SOURCE_DIR.joinpath("build")
TESTS_DIR = ROOT_DIR.joinpath("src", "tests")
PACKAGE_DIR = ROOT_DIR.joinpath("src", "cvops")

sys.path.insert(0, str(TESTS_DIR))


def update_submodules():
    """Updates the git submodules"""
    subprocess.run(["git", "submodule", "update", "--recursive", "--remote", "--init"], cwd=ROOT_DIR, check=True)


CMAKE_CONFIGURE_COMMAND = [
    "cmake",
    "--no-warn-unused-cli",
    "-DCMAKE_BUILD_TYPE:STRING=Debug",
    "-DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE",
    "-DCMAKE_C_COMPILER:FILEPATH=/usr/bin/gcc-8",
    "-S",
    C_SOURCE_DIR,
    "-B",
    C_BUILD_DIR,
    "-G",
    "Unix Makefiles"
]

CMAKE_BUILD_COMMAND = [
    "cmake",
    "--build",
    C_BUILD_DIR,
]

def bootstrap_cmake():
    """Builds the cvops-inference submodule and copies the built library to the to inference directory """
    subprocess.run(CMAKE_CONFIGURE_COMMAND, cwd=C_SOURCE_DIR, check=True)
    subprocess.run(CMAKE_BUILD_COMMAND, cwd=C_SOURCE_DIR, check=True)

    target_lib_dir = pathlib.Path(ROOT_DIR, "src", "cvops", "inference", "lib")

    if not target_lib_dir.exists():
        target_lib_dir.mkdir()

    for so in C_SOURCE_DIR.glob("**/*.so*"):
        print(f"Copying {so} to {target_lib_dir}")
        shutil.copy(so, target_lib_dir)


def run_tests():
    """Runs the tests"""
    print("Inside run tests")
    bootstrap_cmake()
    from tests import run_tests
    run_tests()
    