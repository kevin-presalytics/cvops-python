# This file contains helper functions for the build process
import subprocess
import pathlib
import shutil


ROOT_DIR = pathlib.Path(__file__).parent.parent.parent.absolute()
C_SOURCE_DIR = ROOT_DIR.joinpath("cvops-inference")
C_BUILD_DIR = C_SOURCE_DIR.joinpath("build")

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
    update_submodules()

    subprocess.run(CMAKE_CONFIGURE_COMMAND, cwd=C_SOURCE_DIR, check=True)
    subprocess.run(CMAKE_BUILD_COMMAND, cwd=C_SOURCE_DIR, check=True)

    target_lib_dir = pathlib.Path(ROOT_DIR, "src", "cvops", "inference", "lib")

    if not target_lib_dir.exists():
        target_lib_dir.mkdir()

    for so in C_SOURCE_DIR.glob("**/*.so*"):
        print(f"Copying {so} to {target_lib_dir}")
        shutil.copy(so, target_lib_dir)

# This didnt' work out, because ctypesgen doesn't support C++11
# def generate_ctypes_bindings():
#     """Generates ctypes bindings for the cvops-inference library"""
#     update_submodules()
#     root_dir = pathlib.Path(__file__).parent.parent.parent.absolute()
#     include_dir = pathlib.Path(root_dir, "cvops-inference", "include")
#     target_dir = pathlib.Path(root_dir, "src", "cvops", "inference", "ctypes")

#     header_files = [
#         "cvops_inference.h",
#         "inference_result.h",
#         "inference_request.h",
#         "inference_manager_interface.h"
#     ]

#     if not target_dir.exists():
#         target_dir.mkdir()

#     command_template = ". venv/bin/activate; ctypesgen --cppflags --includes -o {output} {input}"
#     for header in header_files:
#         file = include_dir.joinpath(header)
#         print(f"Generating ctypes bindings for {file}")
#         cmd = command_template.format(output=target_dir.joinpath(header.replace(".h", ".py")), input=file)
#         subprocess.run([cmd], cwd=root_dir, check=True)
