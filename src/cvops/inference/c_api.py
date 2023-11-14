""" Wraps the DLL and provides a pythonic interface to the C API """
import ctypes
import cvops.inference.c_interfaces as _types
import cvops.inference.loader


class CApi(object):
    """ Wrapper class around the calls to the C API """
    loader: 'cvops.inference.loader.DllLoader'

    def __init__(self):
        self.loader = cvops.inference.loader.DllLoader()
        self.loader.load()
        if not self.dll:
            raise Exception("Dll unable to load")  # pylint: disable=broad-exception-raised
        self._bind_function_definitions()

    @property
    def dll(self):
        """ Returns the dll object """
        return self.loader.dll

    def _bind_function_definitions(self):
        self.dll.start_inference_session.argtypes = [_types.c_inference_session_request_p]
        self.dll.start_inference_session.restype = _types.c_i_inference_manager_p
        self.dll.run_inference.argtypes = [
            _types.c_i_inference_manager_p,
            _types.c_inference_request_p,
            _types.c_inference_result_p
        ]
        self.dll.run_inference.restype = None
        self.dll.end_inference_session.argtypes = [_types.c_i_inference_manager_p]
        self.dll.end_inference_session.restype = None
        self.dll.error_message.restype = ctypes.c_char_p
