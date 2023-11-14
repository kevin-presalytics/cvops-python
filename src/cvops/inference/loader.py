# Loader classes to load the C library
import os
import logging
import pathlib
import platform
import typing
import ctypes
import cvops.config


logger = logging.getLogger(__name__)


class DllLoader(object):
    """ Loads the C library """

    DLL_FILE_NAME = "libcvops"

    project_root: pathlib.Path
    library_path: pathlib.Path
    dll_path: pathlib.Path
    system: str
    processor: str
    debug: bool



    def __init__(self, debug: typing.Optional[bool] = None):
        if debug is None:
            self.debug = cvops.config.SETTINGS.debug

        self.system = platform.system()
        self.processor = platform.processor()
        self.project_root = pathlib.Path(__file__).parent.parent.parent.parent
        if self.debug:
            self.library_path = self.project_root.joinpath("cvops-inference", "build")
        else:
            self.library_path = pathlib.Path(__file__).parent.joinpath("lib")
        self.dll_path = self.get_dll_path()
        self.dll = None

    def load(self):
        """ Loads the dll """
        if self.dll is None:
            self.dll = ctypes.cdll.LoadLibrary(self.dll_path)
            if self.debug:
                # Note: Set breakpoints here to debug the c lbirary by attaching to the process
                logger.debug("DLL loaded in process: %s", os.getpid())
                logger.debug("Loaded dll: %s", self.dll_path)
                
            

    def get_file_extension(self):
        """ Returns dll's file extension based on the system """
        if self.system == "Windows":
            return ".dll"
        elif self.system == "Linux":
            return ".so"
        elif self.system == "Darwin":
            return ".dylib"
        else:
            raise Exception("Unsupported system")

    def check_compatbility(self):
        if self.system == "Windows":
            if self.processor == "AMD64":
                return False  # TODO: Add support for Windows
            else:
                return False
        elif self.system == "Linux":
            if self.processor == "x86_64":
                return True
            elif self.processor == "aarch64":
                return False # TODO: Add support for Linux ARM
            else:
                return False
        elif self.system == "Darwin":  # TODO: Add support for MacOS
            if self.processor == "x86_64":
                return False
            else:
                return False
        else:
            raise Exception(f"Unsupported system: System: {self.system}, Processor: {self.processor}")


    def get_dll_path(self):
        if self.check_compatbility():
            dll_path = self.library_path.joinpath(self.DLL_FILE_NAME + self.get_file_extension())
            if not dll_path.exists():
                raise Exception(f"Unable to find dll: {dll_path}")
            return dll_path
        else:
            raise Exception(f"Unsupported system: System: {self.system}, Processor: {self.processor}")
