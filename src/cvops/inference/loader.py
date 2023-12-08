""" Loader classes to load the C library """
import os
import logging
import pathlib
import platform
import typing
import ctypes
import cvops.config


logger = logging.getLogger(__name__)

# One instance of the C library dll per process
__instances__: typing.Dict[int, ctypes.CDLL] = {}


def get_dll_instance() -> typing.Optional[ctypes.CDLL]:
    """ Returns the C Library dll singleton """
    pid: int = os.getpid()
    if pid not in __instances__:
        return None
    return __instances__[pid]


class DllLoader:
    """ Loads the C library """

    DEFAULT_DLL_FILE_NAME = "libcvops"

    project_root: pathlib.Path
    library_path: pathlib.Path
    dll_path: pathlib.Path
    system: str
    processor: str
    debug: bool
    pid: int
    dll_file_name: str

    def __init__(self,
                 debug: typing.Optional[bool] = None,
                 dll_file_name: typing.Optional[str] = None,
                 **kwargs  # pylint: disable=unused-argument
                 ) -> None:
        if debug is None:
            self.debug = cvops.config.SETTINGS.debug
        else:
            self.debug = debug
        self.pid = os.getpid()
        self.system = platform.system()
        self.processor = platform.processor()
        self.project_root = pathlib.Path(__file__).parent.parent.parent.parent
        self.dll_file_name = dll_file_name or self.DEFAULT_DLL_FILE_NAME
        if self.debug:
            self.library_path = self.project_root.joinpath("cvops-inference", "build")
        else:
            self.library_path = pathlib.Path(__file__).parent.joinpath("lib")
        self.dll_path = self.get_dll_path()

    @property
    def dll(self):
        """ Returns the dll object """
        return get_dll_instance()

    @dll.setter
    def dll(self, value):
        """ Sets the dll object """
        pid = os.getpid()
        __instances__[pid] = value

    def load(self) -> bool:
        """ Loads the dll, return bool indicating if the dll was loaded.  False indicates the dll was already loaded """
        if self.dll is None:
            self.dll = ctypes.cdll.LoadLibrary(str(self.dll_path))
            if self.debug:
                logger.debug("DLL loaded in process: %s", os.getpid())
                # Note: Set breakpoints here to debug the c lbirary by attaching to the process
                logger.debug("Loaded dll: %s", self.dll_path)
            return True
        return False

    def get_file_extension(self):
        """ Returns dll's file extension based on the system """
        if self.system == "Windows":
            return ".dll"
        if self.system == "Linux":
            return ".so"
        if self.system == "Darwin":
            return ".dylib"
        raise RuntimeError("Unsupported system")

    def check_compatbility(self):  # python: disable=R0911
        """ Checks if the system and processor are supported by this library"""
        if self.system == "Windows":
            if self.processor == "AMD64":
                return False  # TODO: Add support for Windows
            return False
        if self.system == "Linux":
            if self.processor == "x86_64":
                return True
            if self.processor == "aarch64":
                return False  # TODO: Add support for Linux ARM
            return False
        if self.system == "Darwin":  # TODO: Add support for MacOS
            if self.processor == "x86_64":
                return False
            return False
        raise Exception(f"Unsupported system: System: {self.system}, Processor: {self.processor}")  # pylint: disable=W0719

    def get_dll_path(self):
        """ Returns the path to the dll"""
        if self.check_compatbility():
            dll_path = self.library_path.joinpath(self.dll_file_name + self.get_file_extension())
            if not dll_path.exists():
                raise Exception(f"Unable to find dll: {dll_path}")  # pylint: disable=W0719
            return dll_path
        raise Exception(f"Unsupported system: System: {self.system}, Processor: {self.processor}")  # pylint: disable=W0719
