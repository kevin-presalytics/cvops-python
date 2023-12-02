""" Wraps the DLL and provides a pythonic interface to the C API """
import ctypes
import cvops.schemas
import cvops.inference.c_interfaces as _types
import cvops.inference.loader


class CApi(cvops.schemas.CooperativeBaseClass):
    """ Wrapper class around the calls to the C API """
    loader: 'cvops.inference.loader.DllLoader'

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.loader = cvops.inference.loader.DllLoader(**kwargs)
        self.loader.load()
        if not self.dll:
            raise Exception("Dll unable to load")  # pylint: disable=broad-exception-raised
        #  if not is_loaded:
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
        ]
        self.dll.run_inference.restype = _types.c_inference_result_p
        self.dll.end_inference_session.argtypes = [_types.c_i_inference_manager_p]
        self.dll.end_inference_session.restype = None
        self.dll.error_message.restype = ctypes.c_char_p
        self.dll.set_color_palette.argtypes = [ctypes.c_char_p]
        self.dll.set_color_palette.restype = None
        self.dll.free_color_palette.argtypes = []
        self.dll.free_color_palette.restype = None
        self.dll.dispose_inference_result.argtypes = [_types.c_inference_result_p]
        self.dll.dispose_inference_result.restype = None
        self.dll.render_inference_result.argtypes = [
            _types.c_inference_result_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.dll.render_inference_result.restype = None
        self.dll.create_tracker.argtypes = None
        self.dll.create_tracker.restype = _types.c_tracker_p
        self.dll.track_image.argtypes = [
            _types.c_tracker_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.dll.track_image.restype = None
        self.dll.dispose_tracker.argtypes = [_types.c_tracker_p]
        self.dll.dispose_tracker.restype = None
        self.dll.update_tracker.argtypes = [
            _types.c_tracker_p,
            _types.c_inference_result_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.dll.update_tracker.restype = None
        self.dll.get_tracker_state.argtypes = [_types.c_tracker_p]
        self.dll.get_tracker_state.restype = _types.c_inference_request_p
