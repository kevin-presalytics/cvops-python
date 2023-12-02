""" This file contains helper functions for the build processes and developer workflows """
import subprocess
import pathlib
import shutil
import sys
import logging


logger = logging.getLogger(__name__)


ROOT_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()
C_SOURCE_DIR = ROOT_DIR.joinpath("cvops-inference")
C_BUILD_DIR = ROOT_DIR.joinpath("build")
TESTS_DIR = ROOT_DIR.joinpath("src", "tests")
PACKAGE_DIR = ROOT_DIR.joinpath("src", "cvops")
CMAKE_EXEC = ROOT_DIR.joinpath("venv", "bin", "cmake")

sys.path.insert(0, str(TESTS_DIR))


def update_submodules():
    """Updates the git submodules"""
    subprocess.run(["git", "submodule", "update", "--recursive", "--remote", "--init"], cwd=ROOT_DIR, check=True)


CMAKE_CONFIGURE_COMMAND = [
    str(CMAKE_EXEC),
    "-G",
    "Ninja",
    "--no-warn-unused-cli",
    "-DCMAKE_BUILD_TYPE:STRING=Debug",
    "-DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE",
    "-S",
    str(C_SOURCE_DIR),
    "-B",
    str(C_BUILD_DIR),
]

CMAKE_BUILD_COMMAND = [
    str(CMAKE_EXEC),
    "--build",
    str(C_BUILD_DIR),
]

CMAKE_CLEAN_COMMAND = [
    str(CMAKE_EXEC),
    "--build",
    str(C_BUILD_DIR),
    "--target",
    "clean",
]

def clean_cmake():
    """Cleans the cmake build directory"""
    try:
        subprocess.run(CMAKE_CLEAN_COMMAND, cwd=C_SOURCE_DIR, check=True)
    except FileNotFoundError as f_err:
        message = "Invalid path to C library source directory.  Did you install the project in editable mode using `pip install -e .[dev]`?"
        raise RuntimeError(message) from f_err
    except Exception as ex:  # pylint: disable=broad-exception-caught
        logger.exception(ex, "Unable to clean C library build directory")


def bootstrap_cmake():
    """Builds the cvops-inference submodule and copies the built library to the to inference directory """
    try:
        subprocess.run(CMAKE_CONFIGURE_COMMAND, cwd=C_SOURCE_DIR, check=True)
        subprocess.run(CMAKE_BUILD_COMMAND, cwd=C_SOURCE_DIR, check=True)
    except FileNotFoundError as f_err:
        message = "Invalid path to C library source directory.  Did you install the project in editable mode using `pip install -e .[dev]`?"
        raise RuntimeError(message) from f_err
    except Exception as ex:  # pylint: disable=broad-exception-caught
        logger.exception(ex, "Unable to build C library")

    target_lib_dir = pathlib.Path(ROOT_DIR, "src", "cvops", "inference", "lib")

    if not target_lib_dir.exists():
        target_lib_dir.mkdir()

    for so in C_SOURCE_DIR.glob("**/*.so*"):
        print(f"Copying {so} to {target_lib_dir}")
        shutil.copy(so, target_lib_dir)

def run_tests():
    """ Runs all tests """
    try:
        bootstrap_cmake()
        from tests import test_all  # pylint: disable=import-outside-toplevel
        test_all()
    except Exception as ex:  # pylint: disable=broad-except
        logger.exception(ex, "Unable to run tests")
        sys.exit(1)


PRE_COMMIT_FILE_CONTENTS = """#!/bin/sh
. venv/bin/activate
python -m autopep8 ./src
run_tests
"""


PRE_COMMIT_FILE_PATH = ROOT_DIR.joinpath(".git", "hooks", "pre-commit")


def install_pre_commit_hooks():
    """ Installs the pre-commit hooks """
    with open(PRE_COMMIT_FILE_PATH, "w", encoding='utf-8') as pre_commit_file:
        pre_commit_file.write(PRE_COMMIT_FILE_CONTENTS)
    PRE_COMMIT_FILE_PATH.chmod(0o755)


def install_hooks():
    """ Installs the pre-commit hooks """
    install_pre_commit_hooks()

def clean():
    """ Cleans the build directories """
    clean_cmake()
